"""Collection of discriminator symbols.

A generator takes random inputs and generate
"""
import numpy as np
import mxnet as mx

from .ops import deconv2d_bn_relu, deconv2d_act

def netD(opt):
    if opt.isTrain:
        use_sigmoid = opt.no_lsgan
        netD = define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                      opt.which_model_netD,
                                      opt.n_layers_D, opt.norm, use_sigmoid)
        return netD

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False):
    # netD = None
    # use_gpu = len(gpu_ids) > 0
    # norm_layer = get_norm_layer(norm_type=norm)

    # if use_gpu:
    #     assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    # if use_gpu:
    #     netD.cuda(device_id=gpu_ids[0])
    # netD.apply(weights_init)
    return netD

# Defines the PatchGAN discriminator with the specified arguments.
def NLayerDiscriminator(input_nc, ndf=64, bn_mom=0.9, n_layers=3, use_sigmoid=False, data=None):

        kw = 4
        padw = int(np.ceil((kw-1)/2))

        data = mx.sym.Variable("data") if data is None else data

        body = mx.sym.Convolution(data=data, num_filter=ndf, kernel=(kw, kw), stride=(2,2),
                                  pad=(padw, padw))
        body = mx.sym.LeakyReLU(data=body, slope=0.2)
        # sequence = [
        #     nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
        #     nn.LeakyReLU(0.2, True)
        # ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            body = mx.sym.Convolution(data=body, num_filter=ndf * nf_mult, kernel=(kw, kw),
                                  	  stride=(2,2), pad=(padw, padw))
            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom)
            body = mx.sym.LeakyReLU(data=body, slope=0.2)
            # sequence += [
            #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            #                     kernel_size=kw, stride=2, padding=padw),
            #     # TODO: use InstanceNorm
            #     norm_layer(ndf * nf_mult, affine=True),
            #     nn.LeakyReLU(0.2, True)
            # ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        body = mx.sym.Convolution(data=body, num_filter=ndf * nf_mult, kernel=(kw, kw),
                              	  stride=(1,1), pad=(padw, padw))
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom)
        body = mx.sym.LeakyReLU(data=body, slope=0.2)
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
        #                     kernel_size=kw, stride=1, padding=padw),
        #     # TODO: useInstanceNorm
        #     norm_layer(ndf * nf_mult, affine=True),
        #     nn.LeakyReLU(0.2, True)
        # ]

        body = mx.sym.Convolution(data=body, num_filter=ndf * nf_mult, kernel=(kw, kw),
                                  stride=(1,1), pad=(padw, padw))
        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            body = mx.sym.Sigmoid(data=body)
            # sequence += [nn.Sigmoid()]

        # self.model = nn.Sequential(*sequence)

        return body
