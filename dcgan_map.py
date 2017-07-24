import logging
import numpy as np
import mxnet as mx
import sys
from tensorboard import summary
from tensorboard import FileWriter
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from mxgan_mine import module, generator, discriminator, viz
from util.visualizer import Visualizer


# Load dataset (from pix2pix code)
opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
dataset_size = data_loader.dataset.dataset_len
print('#training images = %d' % dataset_size)

# TensorBoard logging file
logdir = './logs/'
summary_writer = FileWriter(logdir)

# Build model
cond_data_shape = (opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
data_shape = (opt.batchSize, opt.input_nc+opt.input_nc, opt.fineSize, opt.fineSize)
context = mx.cpu()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')
sym_gen = generator.netG(opt)
sym_dis = discriminator.netD(opt)
gmod = module.GANModule(
    sym_gen,
    sym_dis,
    context=context,
    data_shape=data_shape,
    cond_data_shape=cond_data_shape)

init_normal = mx.init.Normal(sigma=0.02)
init_normal_one = mx.sym.random_normal(loc=1, scale=0.03)
init_zero = mx.init.Zero()
init_wt = mx.init.Mixed(patterns=['batchnorm*gamma', 'batchnorm*beta', '.*'], initializers = [init_normal_one, init_zero, init_normal])
# gmod.modG.init_params(mx.init.Normal(0.02))
gmod.modG.init_params(init_wt)
# gmod.modD.init_params(mx.init.Normal(0.02))
gmod.modD.init_params(init_wt)

gmod.init_optimizer(
    optimizer="adam",
    optimizer_params={
        "learning_rate": 0.0002,
        "wd": 0.,
        "beta1": 0.999,
})

visualizer = Visualizer(opt)

for epoch in range(opt.niter):
    data_loader.dataset.reset()
    for t, batch in enumerate(data_loader.dataset):
        gmod.update(batch)

        temp = gmod.D_real.get() + gmod.D_fake.get()
        s = summary.scalar('ferr', temp[1])
        summary_writer.add_summary(s, t)

        if t % 1 == 0:
            gmod.D_real.reset()
            gmod.D_fake.reset()
            gmod.G_GAN.reset()
            gmod.G_L1.reset()
            print temp[1]
            logging.info("epoch: %d, iter %d, metric=%s", epoch, t, temp)

            visualizer.display_current_results(gmod.get_current_visuals(), epoch)
            # viz.imshow("gout", gmod.temp_outG[0].asnumpy() + 0.5 , 2, flip=True)
            # diff = gmod.temp_diffD[0].asnumpy()
            # diff = (diff - diff.mean()) / diff.std() + 0.5
            # viz.imshow("diff", diff, flip=True)
            # viz.imshow("data", batch.data[0].asnumpy() + 0.5, 2, flip=True)

# close summary_writer
summary_writer.close()
