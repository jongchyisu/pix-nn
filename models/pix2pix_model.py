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


def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))



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
            self.criterionL1 = gluon.loss.L1Loss()

            init_normal = mx.initializer.Normal(sigma=0.02)
            init_normal_one = mx.sym.random_normal(loc=1, scale=0.02)
            init_zero = mx.initializer.Zero()
            init_one = mx.initializer.One()
            ## TODO: can't feed Mixed into gluon for initialization
            ## right now just hard-code initialize in parameter.py ...
            # init_wt = mx.initializer.Mixed(['gamma', 'beta', '.*'], [init_normal_one, init_zero, init_normal])
            init_wt = [init_one, init_zero, init_normal]
            self.netG.collect_params().initialize(init_wt, ctx=opt.context)
            self.netD.collect_params().initialize(init_wt, ctx=opt.context)

            # self.netG.collect_params().initialize(init_normal, ctx=opt.context)
            # self.netD.collect_params().initialize(init_normal, ctx=opt.context)

            # initialize optimizers
            self.optimizer_G = gluon.Trainer(self.netG.collect_params(), 'Adam',
                                             {'learning_rate':opt.lr, 'beta1':0.999})
            self.optimizer_D = gluon.Trainer(self.netD.collect_params(), 'Adam',
                                             {'learning_rate':opt.lr, 'beta1':0.999})

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
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
            # self.loss_D_fake.backward()
            # grad_D_fake = {}
            # for i, param in enumerate(self.optimizer_D._params):
            #     grad_D_fake[param.name] = [grad.copyto(grad.context) for grad in param.list_grad()]

            # for name, param in self.netD.collect_params().items():
            #     print name, param
            #     for grad in param.list_grad():
            #         print grad
            # self.grad_D_fake = [[grad.copyto(grad.context) for grad in param.list_grad()] ]

            # Real
            real_AB = mx.ndarray.concat(self.real_A, self.real_B, dim=1)
            self.pred_real = self.netD.forward(real_AB)
            self.loss_D_real = self.criterionGAN(self.pred_real, True)
            # self.loss_D_real.backward()
            # for i, param in enumerate(self.optimizer_D._params):
            #     for real_grad, fake_grad in zip(param.list_grad(), grad_D_fake[param.name]):
            #         real_grad += fake_grad

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
            # print 'after:', [[np.sum(np.abs(grad.asnumpy())) for grad in param.list_grad()] for param in self.optimizer_G._params]
            # exit(0)

    def optimize_parameters(self):
        self.forward()

        # self.optimizer_D.zero_grad()
        for param in self.optimizer_D._params:
            # print param
            param.zero_grad()
        self.backward_D()
        self.optimizer_D.step(self.batchSize)

        # self.optimizer_G.zero_grad()
        for param in self.optimizer_G._params:
            param.zero_grad()
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
