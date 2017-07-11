"""Collection of generators symbols.

A generator takes random inputs and generate
"""
import numpy as np
import mxnet as mx

from .ops import deconv2d_bn_relu, deconv2d_act


def netG(opt):
    """DCGAN that generates 32x32 images."""
    # code = mx.sym.Variable("code") if code is None else code
    # net = mx.sym.FullyConnected(code, name="g1", num_hidden=4*4*ngf*4, no_bias=True)
    # net = mx.sym.Activation(net, name="gact1", act_type="relu")
    # # 4 x 4
    # net = mx.sym.Reshape(net, shape=(-1, ngf * 4, 4, 4))
    # # 8 x 8
    # net = deconv2d_bn_relu(
    #     net, ishape=(ngf * 4, 4, 4), oshape=(ngf * 2, 8, 8), kshape=(4, 4), prefix="g2")
    # # 16x16
    # net = deconv2d_bn_relu(
    #     net, ishape=(ngf * 2, 8, 8), oshape=(ngf, 16, 16), kshape=(4, 4), prefix="g3")
    # # 32x32
    # net = deconv2d_act(
    #     net, ishape=(ngf, 16, 16), oshape=oshape[-3:], kshape=(4, 4), prefix="g4", act_type=final_act)
    # return net
    netG = define_G(opt.input_nc, opt.output_nc, opt.ngf,
                    opt.which_model_netG, opt.norm, opt.use_dropout)
    return netG


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False):
    # netG = None
    # use_gpu = len(gpu_ids) > 0
    # not supporting instance norm now ...
    # norm_layer = get_norm_layer(norm_type=norm)

    # if use_gpu:
    #     assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        # netG = resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False)
        netG = ResnetGenerator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=9)#, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=6)#, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        # TODO
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, use_dropout=use_dropout)#, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        # TODO
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, use_dropout=use_dropout)#, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    # if len(gpu_ids) > 0:
    #     netG.cuda(device_id=gpu_ids[0])
    # netG.apply(weights_init)
    return netG


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
def ResnetGenerator(input_nc, output_nc, ngf=64, bn_mom=0.9, workspace=256, use_dropout=False, n_blocks=6, code=None):
    assert(n_blocks >= 0)

    code = mx.sym.Variable("code") if code is None else code

    body = mx.sym.Convolution(data=code, num_filter=ngf, kernel=(7, 7), pad=(3, 3),
                                  name='conv0', workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        body = mx.sym.Convolution(data=body, num_filter=ngf*mult*2, kernel=(3, 3), stride=(2, 2),
                                  pad=(1, 1), workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom)
        body = mx.sym.Activation(data=body, act_type='relu')

    mult = 2**n_downsampling
    for i in range(n_blocks):
        body = residual_unit(body, ngf * mult, 'stage%d' %(i+1), use_dropout=use_dropout)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)

        body = mx.sym.Convolution(data=body, num_filter=int(ngf * mult / 2), kernel=(3, 3), stride=(2, 2),
                                  pad=(1, 1), workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom)
        body = mx.sym.Activation(data=body, act_type='relu')

    body = mx.sym.Convolution(data=body, num_filter=output_nc, kernel=(7, 7), pad=(3, 3),
                                  workspace=workspace)
    body = mx.sym.Activation(data=body, act_type='tanh')

    return body



# Define a resnet block
def residual_unit(data, dim, name, stride=(1,1), bn_mom=0.9, workspace=256, use_dropout=False):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=dim, kernel=(3,3), stride=stride, pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv1')
    if use_dropout:
        conv1 = mx.symbol.Dropout(data=conv1, p=0.5, name=name + '_do1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=dim, kernel=(3,3), stride=(1,1), pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    # if dim_match:
    shortcut = data
    # else:
    #     shortcut = mx.sym.Convolution(data=act1, num_filter=dim, kernel=(1,1), stride=stride, no_bias=True,
                                        # workspace=workspace, name=name+'_sc')
    # if memonger:
    #     shortcut._set_attr(mirror_stage='True')
    return conv2 + shortcut

