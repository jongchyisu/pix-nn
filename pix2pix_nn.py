import time
import logging
import numpy as np
import mxnet as mx
import sys
#from tensorboard import summary
#from tensorboard import FileWriter
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from data.mxnet_iter import AlignedIter

# Load dataset (from pix2pix code)
opt = TrainOptions().parse()
if opt.gpu_ids == []:
    opt.context = mx.cpu()
else:
    opt.context = [mx.gpu(gpu_ids) for gpu_ids in opt.gpu_ids]

# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
# dataset_size = data_loader.dataset.dataset_len
data_iter = AlignedIter(opt)
opt.dataset_size = data_iter.dataset_len
print('#training images = %d' % opt.dataset_size)

# # TensorBoard logging file
# logdir = './logs/'
# summary_writer = FileWriter(logdir)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

# Build model
model = create_model(opt)
visualizer = Visualizer(opt)

# Training
total_steps = 0
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    data_iter.reset()

    for data in data_iter:
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - opt.dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/opt.dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        # lr linear decay, implemented in lr_scheduler which is fed into optimizer
        print 'lr_G: ', self.optimizer_G._optimizer._get_lr
        print 'lr_D: ', self.optimizer_D._optimizer._get_lr

# # close summary_writer
# summary_writer.close()
