import os.path
import random
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import mxnet as mx
import cv2

class AlignedIter(mx.io.DataIter):
    def __init__(self, opt):
        # self._provide_data = zip(data_names, data_shapes)
        # self._provide_label = zip(label_names, label_shapes)
        # self.num_batches = num_batches
        # self.data_gen = data_gen
        # self.label_gen = label_gen
        # self.get_image = get_image
        self.cur_batch = 0

        #### pytorch
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        #### mine
        self.dataset_len = len(self.AB_paths)

    def __iter__(self):
        return self

    def reset(self):
        random.shuffle(self.AB_paths)
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        desc = [mx.io.DataDesc('cond_data',(1, 3, 256, 256)),
                mx.io.DataDesc('data',(1, 3, 256, 256))]
        return desc

    @property
    def provide_label(self):
        desc = [mx.io.DataDesc('dloss_label', (1, )),
                mx.io.DataDesc('l1_loss_label', (1, 3, 256, 256))]
        return desc

    def next(self):
        if self.cur_batch < self.dataset_len:
            A,B = self.get_image(self.cur_batch)
            self.cur_batch += 1
            label = None
            # data = [mx.nd.array(g(d[1])) for d,g in zip(self.provide_data, self.get_image(self.cur_batch))]
            # label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            # return mx.io.DataBatch(data, label)
            return mx.io.DataBatch(data=[A,B], label=[label,B])
        else:
            raise StopIteration

    def get_image(self, index):
        AB_path = self.AB_paths[index]

        img = mx.image.imdecode(open(AB_path).read()) # default is RGB
        
        ## resize to w x h
        ## TODO: make sure it's bicubic
        img = mx.image.imresize(img, self.opt.loadSize * 2, self.opt.loadSize, interp = cv2.INTER_CUBIC)
        
        # convert to [0,1] then normalize
        img = img.astype('float32')
        img /= 255.0
        AB = mx.image.color_normalize(img, 0.5, 0.5)

        ## crop a random w x h region from image
        # tmp, coord = mx.image.random_crop(img, (150, 200))
        # print(coord)
        # plt.imshow(tmp.asnumpy()); plt.show()

        # separate A and B images
        w_total = AB.shape[1]
        w = int(w_total / 2)
        h = AB.shape[0]
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        tempA = mx.nd.slice_axis(AB, axis=0, begin=h_offset, end=h_offset + self.opt.fineSize)
        A = mx.nd.slice_axis(tempA, axis=1, begin=w_offset, end=w_offset + self.opt.fineSize)
        tempB = mx.nd.slice_axis(AB, axis=0, begin=h_offset, end=h_offset + self.opt.fineSize)
        B = mx.nd.slice_axis(tempB, axis=1, begin=w + w_offset, end=w + w_offset + self.opt.fineSize)

        # flipping
        if (not self.opt.no_flip) and random.random() < 0.5:
            A = mx.nd.reverse(A, axis=1)
            B = mx.nd.reverse(B, axis=1)

        # change to BCWH format
        A = mx.nd.transpose(A, (2, 0, 1))
        # A = mx.nd.swapaxes(A, 0, 2)
        # A = mx.nd.swapaxes(A, 1, 2)
        A = mx.nd.expand_dims(A, axis=0)
        B = mx.nd.transpose(B, (2, 0, 1))
        # B = mx.nd.swapaxes(B, 0, 2)
        # B = mx.nd.swapaxes(B, 1, 2)
        B = mx.nd.expand_dims(B, axis=0)

        return A,B
        # return {'A': A, 'B': B,'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
