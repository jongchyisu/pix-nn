import os.path
import random
# import torchvision.transforms as transforms
#import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import mxnet as mx

class AlignedDataset(BaseDataset):

    def initialize(self, opt):
        # self._provide_data = zip(data_names, data_shapes)
        # self._provide_label = zip(label_names, label_shapes)
        #self.num_batches = num_batches
        # self.data_gen = data_gen
        # self.label_gen = label_gen
        #self.get_image = get_image
        self.cur_batch = 0

        #### pytorch
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        #### mine
        self.dataset_len = len(self.AB_paths)

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        desc = [mx.io.DataDesc('code',(1, 3, 256, 256)),
                mx.io.DataDesc('data',(1, 3, 256, 256))]
        return desc

    # @property
    # def provide_label(self):
    #     return self._provide_label

    def next(self):
        if self.cur_batch < self.dataset_len:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self.provide_data, self.get_image(self.cur_batch))]
            # label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            # return mx.io.DataBatch(data, label)
            return mx.io.DataBatch(data)
        else:
            raise StopIteration

    #### pytorch
    # def initialize(self, opt):
    #     self.opt = opt
    #     self.root = opt.dataroot
    #     self.dir_AB = os.path.join(opt.dataroot, opt.phase)

    #     self.AB_paths = sorted(make_dataset(self.dir_AB))

    #     assert(opt.resize_or_crop == 'resize_and_crop')

    #     transform_list = [transforms.ToTensor(),
    #                       transforms.Normalize((0.5, 0.5, 0.5),
    #                                            (0.5, 0.5, 0.5))]

    #     self.transform = transforms.Compose(transform_list)

    def get_image(self, index):
        AB_path = self.AB_paths[index]

        img = mx.image.imdecode(open(AB_path).read()) # default is RGB
        
        # resize to w x h
        img = mx.image.imresize(img, self.opt.loadSize * 2, self.opt.loadSize)

        import pdb
        pdb.set_trace()
        
        AB = mx.image.color_normalize(img, 0.5, 0.5)

        # # crop a random w x h region from image
        # tmp, coord = mx.image.random_crop(img, (150, 200))
        # print(coord)
        # plt.imshow(tmp.asnumpy()); plt.show()

        # pytorch code
        # AB = Image.open(AB_path).convert('RGB')
        # AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        # AB = self.transform(AB)

        import pdb
        pdb.set_trace()

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
