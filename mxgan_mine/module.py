"""Modules for training GAN, work with multiple GPU."""
import mxnet as mx
from . import ops
import numpy as np
import util.util as util
from collections import OrderedDict

class GANBaseModule(object):
    """Base class to hold gan data
    """
    def __init__(self,
                 generator,
                 context,
                 cond_data_shape):
        # generator
        self.modG = mx.mod.Module(symbol=generator,
                                  data_names=('cond_data',),
                                  label_names=('l1_loss_label',),
                                  context=context)
        self.modG.bind(data_shapes=[('cond_data', cond_data_shape)])
        # for visualization
        self.fake_B = None
        self.real_A = None
        self.real_B = None

        self.temp_gradD = None
        self.context = context if isinstance(context, list) else [context]
        self.outputs_fake = None
        self.outputs_real = None

    def _save_temp_gradD(self):
        if self.temp_gradD is None:
            self.temp_gradD = [
                [grad.copyto(grad.context) for grad in grads]
                for grads in self.modD._exec_group.grad_arrays]
        else:
            for gradsr, gradsf in zip(self.modD._exec_group.grad_arrays, self.temp_gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr.copyto(gradf)

    def _add_temp_gradD(self):
        # add back saved gradient
        for gradsr, gradsf in zip(self.modD._exec_group.grad_arrays, self.temp_gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf

    def init_params(self, *args, **kwargs):
        self.modG.init_params(*args, **kwargs)
        self.modD.init_params(*args, **kwargs)

    def init_optimizer(self, *args, **kwargs):
        self.modG.init_optimizer(*args, **kwargs)
        self.modD.init_optimizer(*args, **kwargs)


class GANModule(GANBaseModule):
    """A thin wrapper of module to group generator and discriminator together in GAN.

    Example
    -------
    lr = 0.0005
    mod = GANModule(generator, encoder, context=mx.gpu()),
    mod.bind(data_shape=(3, 32, 32))
    mod.init_params(mx.init.Xavier())
    mod.init_optimizer("adam", optimizer_params={
        "learning_rate": lr,
    })

    for t, batch in enumerate(train_data):
        mod.update(batch)
        # update metrics
        mod.temp_label[:] = 0.0
        metricG.update_metric(mod.outputs_fake, [mod.temp_label])
        mod.temp_label[:] = 1.0
        metricD.update_metric(mod.outputs_real, [mod.temp_label])
        # visualize
        if t % 100 == 0:
            gen_image = mod.temp_outG[0].asnumpy()
            gen_diff = mod.temp_diffD[0].asnumpy()
            viz.imshow("gen_image", gen_image)
            viz.imshow("gen_diff", gen_diff)
    """
    def __init__(self,
                 generator,
                 discriminator,
                 context,
                 data_shape,
                 cond_data_shape,
                 pos_label=0.9):
        super(GANModule, self).__init__(
            generator, context, cond_data_shape)
        context = context if isinstance(context, list) else [context]
        self.batch_size = data_shape[0]
        label_shape = (self.batch_size, )
        ## moved into discriminator
        # discriminator = mx.sym.FullyConnected(discriminator, num_hidden=1, name='fc_dloss')
        # self.loss_GAN = mx.sym.LogisticRegressionOutput(discriminator, name='dloss')
        self.modD = mx.mod.Module(symbol=discriminator,
                                  data_names=('data',),
                                  label_names=('dloss_label',),
                                  context=context)
        self.modD.bind(data_shapes=[('data', data_shape)],
                       label_shapes=[('dloss_label', label_shape)],
                       inputs_need_grad=True)
        self.pos_label = pos_label
        self.temp_label = mx.nd.zeros(
            label_shape, ctx=context[-1])

        # define loss functions
        def ferr(label, pred):
            pred = pred.ravel()
            label = label.ravel()
            return np.abs(label - (pred > 0.5)).sum() / label.shape[0]
            #return ((pred > 0.5) == label).mean()

        # DCGAN uses entropy??
        def fentropy(label, pred):
            pred = pred.ravel()
            label = label.ravel()
            return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()

        self.loss_GAN = mx.sym.Variable('dloss')
        self.loss_G_L1 = mx.sym.Variable('l1_loss')
        self.loss_G_all = mx.sym.Group([self.loss_GAN, self.loss_G_L1])

        self.D_real = mx.metric.CustomMetric(ferr)
        self.D_fake = mx.metric.CustomMetric(ferr)
        self.G_GAN = mx.metric.CustomMetric(ferr)
        self.G_L1 = mx.metric.MAE()

    def update(self, batch):
        self.real_A = batch.data[0]
        self.real_B = batch.data[1]
        """Update the model for a single batch."""
        # generate fake image
        self.modG.forward(mx.io.DataBatch([self.real_A], [self.real_B]), is_train=True)
        self.fake_B = self.modG.get_outputs()[0]
        self.fake_B_l1 = self.modG.get_outputs()[1]
        self.fake_B_l1_loss = mx.ndarray.mean(data=self.fake_B_l1+1.0, axis=[1,2,3])*128
        
        # feed fake_AB into Discriminator
        self.temp_label[:] = 0
        fake_AB = mx.ndarray.concat(self.real_A, self.fake_B, dim=1)
        self.modD.forward(mx.io.DataBatch([fake_AB], [self.temp_label]), is_train=True)
        self.modD.backward()
        self._save_temp_gradD()

        self.modD.update_metric(self.D_fake, [self.temp_label])
        
        # feed real_AB into Discriminator
        self.temp_label[:] = 1
        real_AB = mx.ndarray.concat(self.real_A, self.real_B, dim=1)
        self.modD.forward(mx.io.DataBatch([real_AB], [self.temp_label]), is_train=True)
        self.modD.backward()

        self.modD.update_metric(self.D_real, [self.temp_label])

        # update discriminator
        self._add_temp_gradD()
        self.modD.update()
        self.outputs_real = self.modD.get_outputs()

        # update generator
        self.temp_label[:] = 1
        self.modD.forward(mx.io.DataBatch([fake_AB], [self.temp_label]), is_train=True)
        pred_fake = self.modD.get_outputs()

        # self.G_L1.update(data, fake_B[0])
        self.modD.backward()
        diffD = self.modD.get_input_grads()
        
        diffD_slice = mx.ndarray.slice_axis(diffD[0], axis=1, begin=0, end=3)
        self.modG.backward([self.fake_B_l1, diffD_slice])
        self.modG.update()

        # self.modG.update_metric(self.G_GAN, [self.temp_label])
        # self.G_L1.update(data, fake_B[0])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A)
        fake_B = util.tensor2im(self.fake_B)
        real_B = util.tensor2im(self.real_B)
        # import pdb
        # pdb.set_trace()
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])


# class SemiGANModule(GANBaseModule):
#     """A semisupervised gan that can take both labeled and unlabeled data.
#     """
#     def __init__(self,
#                  symbol_generator,
#                  symbol_encoder,
#                  context,
#                  data_shape,
#                  code_shape,
#                  num_class,
#                  pos_label=0.9):
#         super(SemiGANModule, self).__init__(
#             symbol_generator, context, code_shape)
#         # the discriminator encoder
#         context = context if isinstance(context, list) else [context]
#         batch_size = data_shape[0]
#         self.num_class = num_class
#         encoder = symbol_encoder
#         encoder = mx.sym.FullyConnected(
#             encoder, num_hidden=num_class + 1, name="energy")
#         self.modD = mx.mod.Module(symbol=encoder,
#                                   data_names=("data",),
#                                   label_names=None,
#                                   context=context)
#         self.modD.bind(data_shapes=[("data", data_shape)],
#                        inputs_need_grad=True)
#         self.pos_label = pos_label
#         # discriminator loss
#         energy = mx.sym.Variable("energy")
#         label_out = mx.sym.SoftmaxOutput(energy, name="softmax")
#         ul_pos_energy = mx.sym.slice_axis(
#             energy, axis=1, begin=0, end=num_class)
#         ul_pos_energy = ops.log_sum_exp(
#             ul_pos_energy, axis=1, keepdims=True, name="ul_pos")
#         ul_neg_energy = mx.sym.slice_axis(
#             energy, axis=1, begin=num_class, end=num_class + 1)
#         ul_pos_prob = mx.sym.LogisticRegressionOutput(
#             ul_pos_energy - ul_neg_energy, name="dloss")
#         # use module to bind the
#         self.mod_label_out = mx.mod.Module(
#             symbol=label_out,
#             data_names=("energy",),
#             label_names=("softmax_label",),
#             context=context)
#         self.mod_label_out.bind(
#             data_shapes=[("energy", (batch_size, num_class + 1))],
#             label_shapes=[("softmax_label", (batch_size,))],
#             inputs_need_grad=True)
#         self.mod_ul_out = mx.mod.Module(
#             symbol=ul_pos_prob,
#             data_names=("energy",),
#             label_names=("dloss_label",),
#             context=context)
#         self.mod_ul_out.bind(
#             data_shapes=[("energy", (batch_size, num_class + 1))],
#             label_shapes=[("dloss_label", (batch_size,))],
#             inputs_need_grad=True)
#         self.mod_ul_out.init_params()
#         self.mod_label_out.init_params()
#         self.temp_label = mx.nd.zeros(
#             (batch_size,), ctx=context[0])

#     def update(self, dbatch, is_labeled):
#         """Update the model for a single batch."""
#         # generate fake image
#         mx.random.normal(0, 1.0, out=self.temp_rbatch.data[0])
#         self.modG.forward(self.temp_rbatch)
#         outG = self.modG.get_outputs()
#         self.temp_label[:] = self.num_class
#         self.modD.forward(mx.io.DataBatch(outG, []), is_train=True)
#         self.mod_label_out.forward(
#             mx.io.DataBatch(self.modD.get_outputs(), [self.temp_label]), is_train=True)
#         self.mod_label_out.backward()
#         self.modD.backward(self.mod_label_out.get_input_grads())
#         self._save_temp_gradD()
#         # update generator
#         self.temp_label[:] = 1
#         self.modD.forward(mx.io.DataBatch(outG, []), is_train=True)
#         self.mod_ul_out.forward(
#             mx.io.DataBatch(self.modD.get_outputs(), [self.temp_label]), is_train=True)
#         self.mod_ul_out.backward()
#         self.modD.backward(self.mod_ul_out.get_input_grads())
#         diffD = self.modD.get_input_grads()
#         self.modG.backward(diffD)
#         self.modG.update()
#         self.outputs_fake = [x.copyto(x.context) for x in self.mod_ul_out.get_outputs()]
#         # update discriminator
#         self.modD.forward(mx.io.DataBatch(dbatch.data, []), is_train=True)
#         outD = self.modD.get_outputs()
#         self.temp_label[:] = self.pos_label
#         self.mod_ul_out.forward(
#             mx.io.DataBatch(outD, [self.temp_label]), is_train=True)
#         self.outputs_real = [x.copyto(x.context) for x in self.mod_ul_out.get_outputs()]
#         if is_labeled:
#             self.mod_label_out.forward(
#                 mx.io.DataBatch(outD, dbatch.label), is_train=True)
#             self.mod_label_out.backward()
#             egrad = self.mod_label_out.get_input_grads()
#         else:
#             self.mod_ul_out.backward()
#             egrad = self.mod_ul_out.get_input_grads()
#         self.modD.backward(egrad)
#         self._add_temp_gradD()
#         self.modD.update()
#         self.temp_outG = outG
#         self.temp_diffD = diffD
