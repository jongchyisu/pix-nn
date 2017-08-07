# import torch
# import torch.nn as nn
# from torch.nn import init
import functools
# from torch.autograd import Variable
import numpy as np
from mxnet.gluon import nn
from mxnet import autograd
import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd, gpu
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        # norm_layer = functools.partial(nn.BatchNorm, affine=True)
        norm_layer = nn.BatchNorm
    elif norm_type == 'instance':
        raise NotImplementedError('gluon has no InstanceNorm yet')
        # norm_layer = functools.partial(nn.InstanceNorm, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, context=mx.cpu()):
    # netG = None
    # use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    # if use_gpu:
    #     assert(torch.cuda.is_available())
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'resnet_1blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=1)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    # if len(gpu_ids) > 0:
    #     netG.cuda(device_id=gpu_ids[0])
    # netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, context=mx.cpu()):
    # netD = None
    # use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    # if use_gpu:
    #     assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    # if use_gpu:
    #     netD.cuda(device_id=gpu_ids[0])
    # netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    print(net)
    print('Total number of parameters: TODO... %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(gluon.Block):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, context=mx.cpu()):
                 # tensor=mx.nd):#tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.context = context
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.use_lsgan = use_lsgan
        # self.Tensor = tensor
        if use_lsgan:
            self.loss = gluon.loss.L1Loss()
        else:
            # nn.BCEloss in pytorch original implementation, fixed weight for now...
            self.loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False,
                        from_logits=True, axis=[-1,-2], weight=1.0/(35*35))

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.size != input.size))
            if create_label:
                self.real_label_var = mx.nd.full(input.shape, self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.size != input.size))
            if create_label:
                self.fake_label_var = mx.nd.full(input.shape, self.fake_label)
            target_tensor = self.fake_label_var

        target_tensor = gluon.utils.split_and_load(target_tensor, self.context)[0]
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        if not self.use_lsgan:
            log_input = mx.nd.log(input)
            log_input_ = mx.nd.log(1.0-input)
            return self.loss(log_input, target_tensor) + self.loss(log_input_, 1.0-target_tensor)
        else:
            return self.loss(input, target_tensor) + self.loss(1.0-input, 1.0-target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(gluon.Block):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = nn.Sequential(prefix='netG_')
        with model.name_scope():
            model.add(nn.pad(pad_width=(0,0,0,0,3,3,3,3), mode='reflect'))
            model.add(nn.Conv2D(ngf, kernel_size=7, padding=0))
            model.add(norm_layer())
            model.add(nn.relu())
            n_downsampling = 2
            for i in range(n_downsampling):
                mult = 2**i
                model.add(nn.Conv2D(ngf * mult * 2, kernel_size=3, stride=(2,2), padding=1))
                model.add(norm_layer())
                model.add(nn.relu())

            mult = 2**n_downsampling
            for i in range(n_blocks):
                model.add(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout))

            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model.add(nn.Conv2DTranspose(channels=int(ngf * mult / 2),
                                             kernel_size=3, strides=(2,2),
                                             padding=1, output_padding=1))
                model.add(norm_layer())
                model.add(nn.relu())
            model.add(nn.pad(pad_width=(0,0,0,0,3,3,3,3), mode='reflect'))
            model.add(nn.Conv2D(output_nc, kernel_size=7, padding=0))
            model.add(nn.tanh())

        # self.model = nn.Sequential(model)
        self.model = model

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(input)


# Define a resnet block
class ResnetBlock(gluon.Block):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = nn.Sequential()
        # conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block.add(nn.pad(pad_width=(0,0,0,0,1,1,1,1), mode='reflect'))
        elif padding_type == 'replicate':
            conv_block.add(nn.pad(pad_width=(0,0,0,0,1,1,1,1), mode='edge'))
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block.add(nn.Conv2D(dim, kernel_size=3, padding=p))
        conv_block.add(norm_layer())
        conv_block.add(nn.relu())
        if use_dropout:
            conv_block.add(nn.Dropout(0.5))

        p = 0
        if padding_type == 'reflect':
            conv_block.add(nn.pad(pad_width=(0,0,0,0,1,1,1,1), mode='reflect'))
        elif padding_type == 'replicate':
            conv_block.add(nn.pad(pad_width=(0,0,0,0,1,1,1,1), mode='edge'))
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block.add(nn.Conv2D(dim, kernel_size=3, padding=p))
        conv_block.add(norm_layer())

        return conv_block
        # return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(gluon.Block):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_ct = 0
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True, unet_ct=unet_ct)
        for i in range(num_downs - 5):
            unet_ct+=1
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout, unet_ct=unet_ct)
        unet_ct+=1
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer, unet_ct=unet_ct)
        unet_ct+=1
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, unet_ct=unet_ct)
        unet_ct+=1
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer, unet_ct=unet_ct)
        unet_ct+=1
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, unet_ct=unet_ct)

        self.model = unet_block

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(gluon.Block):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm, use_dropout=False, unet_ct=0):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        prefix = 'netG_' + str(unet_ct) + '_'
        model = nn.Sequential(prefix=prefix)
        with model.name_scope():
            downconv = nn.Conv2D(inner_nc, kernel_size=4,
                                 stride=(2,2), padding=1)
            downrelu = nn.LeakyReLU(0.2)
            downnorm = norm_layer()
            uprelu = nn.relu()
            upnorm = norm_layer()

            if outermost:
                upconv = nn.Conv2DTranspose(outer_nc,
                                            kernel_size=4, strides=(2,2),
                                            padding=1)
                # down = [downconv]
                # up = [uprelu, upconv, nn.tanh()]
                # model = down + [submodule] + up
                model.add(downconv)
                model.add(submodule)
                model.add(uprelu)
                model.add(upconv)
                model.add(nn.tanh())
            elif innermost:
                upconv = nn.Conv2DTranspose(outer_nc,
                                            kernel_size=4, strides=(2,2),
                                            padding=1)
                # down = [downrelu, downconv]
                # up = [uprelu, upconv, upnorm]
                # model = down + up
                model.add(downrelu)
                model.add(downconv)
                model.add(uprelu)
                model.add(upconv)
                model.add(upnorm)
            else:
                upconv = nn.Conv2DTranspose(outer_nc,
                                            kernel_size=4, strides=(2,2),
                                            padding=1)
                # down = [downrelu, downconv, downnorm]
                # up = [uprelu, upconv, upnorm]
                model.add(downrelu)
                model.add(downconv)
                model.add(downnorm)

                model.add(submodule)

                model.add(uprelu)
                model.add(upconv)
                model.add(upnorm)

                if use_dropout:
                    # model = down + [submodule] + up + [nn.Dropout(0.5)]
                    model.add(nn.Dropout(0.5))
                # else:
                    # model = down + [submodule] + up

        self.model = model
        # self.model = nn.Sequential(model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # return torch.cat([self.model(x), x], 1)
            return mx.nd.concat(self.model(x), x, dim=1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(gluon.Block):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        model = nn.Sequential(prefix='netD_')
        with model.name_scope():
            kw = 4
            padw = int(np.ceil((kw-1)/2.0))
            model.add(nn.Conv2D(ndf, kernel_size=kw, stride=(2,2), padding=padw))
            model.add(nn.LeakyReLU(0.2))

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                model.add(nn.Conv2D(ndf * nf_mult, kernel_size=kw, stride=(2,2), padding=padw))
                model.add(norm_layer())
                model.add(nn.LeakyReLU(0.2))

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)

            model.add(nn.Conv2D(ndf * nf_mult, kernel_size=kw, stride=(1,1), padding=padw))
            model.add(norm_layer())
            model.add(nn.LeakyReLU(0.2))

            model.add(nn.Conv2D(1, kernel_size=kw, stride=(1,1), padding=padw))

            if use_sigmoid:
                model.add(nn.Sigmoid())

        self.model = model
        # self.model = nn.Sequential(sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(input)
