import numpy as np
# import torch
import os
from collections import OrderedDict
# from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import mxnet.gluon as gluon
import mxnet as mx
from mxnet import autograd, gpu


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        self.batchSize = opt.batchSize
        self.context = opt.context

        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = mx.nd.zeros((opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize))
        self.input_B = mx.nd.zeros((opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize))
        # self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
        #                            opt.fineSize, opt.fineSize)
        # self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
        #                            opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.context)
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD, opt.n_layers_D,
                                          opt.norm, opt.context)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, context=self.context)
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = gluon.loss.L1Loss()
            # self.criterionL1 = torch.nn.L1Loss()

            init_normal = mx.init.Normal(sigma=0.02)
            init_normal_one = mx.sym.random_normal(loc=1, scale=0.03)
            init_zero = mx.init.Zero()
            init_wt = mx.init.Mixed(['batchnorm*gamma', 'batchnorm*beta', '.*'], [init_normal_one, init_zero, init_normal])
            # self.netG.collect_params().initialize(init_wt, ctx=context)
            # self.netD.collect_params().initialize(init_wt, ctx=context)
            self.netG.collect_params().initialize(init_normal, ctx=opt.context)
            self.netD.collect_params().initialize(init_normal, ctx=opt.context)

            # initialize optimizers
            self.optimizer_G = gluon.Trainer(self.netG.collect_params(), 'Adam',
                                             {'learning_rate':opt.lr, 'beta1':0.999})
            self.optimizer_D = gluon.Trainer(self.netD.collect_params(), 'Adam',
                                             {'learning_rate':opt.lr, 'beta1':0.999})
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        # self.input_A = input.data[0]
        # self.input_B = input.data[1]
        self.input_A = gluon.utils.split_and_load(input.data[0], self.context)[0]
        self.input_B = gluon.utils.split_and_load(input.data[1], self.context)[0]
        # input_A = input.data[0].as_in_context(context)
        # input_B = input.data[1].as_in_context(context)

        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A']
        # self.input_A.resize_(input_A.size()).copy_(input_A)
        # self.input_B.resize_(input_B.size()).copy_(input_B)

        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        with autograd.record():
            self.real_A = self.input_A
            self.real_A.attach_grad()
            # self.real_A = Variable(self.input_A)
            self.fake_B = self.netG.forward(self.real_A)
            self.real_B = self.input_B
            self.real_B.attach_grad()
            # self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = self.input_A
        # self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = self.input_B
        # self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        print('have not implemented yet')
        return self.image_paths

    def backward_D(self):
        with autograd.record():
            # Fake
            # stop backprop to the generator by detaching fake_B
            fake_AB = mx.ndarray.concat(self.real_A, self.fake_B, dim=1)
            self.pred_fake = self.netD.forward(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

            # Real
            real_AB = mx.ndarray.concat(self.real_A, self.real_B, dim=1)
            self.pred_real = self.netD.forward(real_AB)
            self.loss_D_real = self.criterionGAN(self.pred_real, True)

            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

            self.loss_D.backward()

    def backward_G(self):
        with autograd.record():
            # First, G(A) should fake the discriminator
            fake_AB = mx.ndarray.concat(self.real_A, self.fake_B, dim=1)
            pred_fake = self.netD.forward(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

            self.loss_G = self.loss_G_GAN + self.loss_G_L1

            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step(self.batchSize)

        # self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step(self.batchSize)

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.asnumpy()[0]),
                            ('G_L1', self.loss_G_L1.asnumpy()[0]),
                            ('D_real', self.loss_D_real.asnumpy()[0]),
                            ('D_fake', self.loss_D_fake.asnumpy()[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A)
        fake_B = util.tensor2im(self.fake_B)
        real_B = util.tensor2im(self.real_B)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label)
        self.save_network(self.netD, 'D', label)

    # def update_learning_rate(self):
    #     lrd = self.opt.lr / self.opt.niter_decay
    #     lr = self.old_lr - lrd
    #     for param_group in self.optimizer_D.param_groups:
    #         param_group['lr'] = lr
    #     for param_group in self.optimizer_G.param_groups:
    #         param_group['lr'] = lr
    #     print('update learning rate: %f -> %f' % (self.old_lr, lr))
    #     self.old_lr = lr
