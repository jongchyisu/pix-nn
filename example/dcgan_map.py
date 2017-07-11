import logging
import numpy as np
import mxnet as mx
import sys
from tensorboard import summary
from tensorboard import FileWriter

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader

# Load dataset (from pix2pix code)
opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
dataset_size = data_loader.dataset.dataset_len
print('#training images = %d' % dataset_size)

# TensorBoard logging file
logdir = './logs/'
summary_writer = FileWriter(logdir)

sys.path.append("..")

from mxgan_mine import module, generator, discriminator, viz

def ferr(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return np.abs(label - (pred > 0.5)).sum() / label.shape[0]

ngf= 64
lr = 0.0002
beta1 = 0.5
batch_size = 1
code_shape = (batch_size, 3, 256, 256)
num_epoch = 100
data_shape = (batch_size, 3, 256, 256)
context = mx.cpu()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')
sym_gen = generator.netG(opt)
sym_dis = discriminator.netD(opt)
gmod = module.GANModule(
    sym_gen,
    sym_dis,
    context=context,
    data_shape=data_shape,
    code_shape=code_shape)

gmod.modG.init_params(mx.init.Normal(0.05))
gmod.modD.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))

gmod.init_optimizer(
    optimizer="adam",
    optimizer_params={
        "learning_rate": lr,
        "wd": 0.,
        "beta1": beta1,
})

# Using mxnet iterator
# data_root_dir = 'dataset/maps/train/'
# data_list_dir = 'dataset/maps_train.lst'
# train = mx.image.ImageIter(
#     path_root = data_root_dir,
#     path_imglist = data_list_dir,
#     data_shape = data_shape[1:],
#     batch_size = batch_size,
#     shuffle=True)

metric_acc = mx.metric.CustomMetric(ferr)

for epoch in range(num_epoch):
    # train.reset()
    data_loader.dataset.reset()
    metric_acc.reset()
    for t, batch in enumerate(data_loader.dataset):
        import pdb
        pdb.set_trace()
        batch.data[0] = batch.data[0] * (1.0 / 255.0) - 0.5
        gmod.update(batch)
        gmod.temp_label[:] = 0.0
        metric_acc.update([gmod.temp_label], gmod.outputs_fake)
        gmod.temp_label[:] = 1.0
        metric_acc.update([gmod.temp_label], gmod.outputs_real)

        temp = metric_acc.get()
        s = summary.scalar('ferr', temp[1])
        summary_writer.add_summary(s, t)

        if t % 50 == 0:
            print temp[1]
            logging.info("epoch: %d, iter %d, metric=%s", epoch, t, temp)
            #viz.imshow("gout", gmod.temp_outG[0].asnumpy() + 0.5 , 2, flip=True)
            #diff = gmod.temp_diffD[0].asnumpy()
            #diff = (diff - diff.mean()) / diff.std() + 0.5
            #viz.imshow("diff", diff, flip=True)
            #viz.imshow("data", batch.data[0].asnumpy() + 0.5, 2, flip=True)

# close summary_writer
summary_writer.close()
