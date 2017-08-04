# pix2pix mxnet with gluon and python

Follow the instructions from the original [pix2pix github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to download the dataset under 'dataset/'

Run the training code like this:
```
python pix2pix_nn.py  --dataroot dataset/maps --name maps_pix2pix_unet_256 --which_model_netG unet_256 --gpu_ids 1 --lambda_A 100 --print_freq 100 --display_freq 100 --port 8098
```

Open visdom server:
```
python -m visdom.server
```

Then you can got to localhost:8098 to see the training curves and results.


### [Note:] Need to add following lines in 'gluon/nn/basic_layers.py'

```python
class pad(HybridBlock):
    def __init__(self, pad_width, mode, **kwargs):
        super(pad, self).__init__(**kwargs)
        self._pad_width = pad_width
        self._mode = mode

    def hybrid_forward(self, F, x):
        return F.pad(x, pad_width=self._pad_width, mode=self._mode)

    def __repr__(self):
        s = '{name}({pad_width}, {mode})'
        return s.format(name=self.__class__.__name__,
                        pad_width=self._pad_width,
                        mode=self._mode)


class relu(HybridBlock):
    def __init__(self, **kwargs):
        super(relu, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.relu(x)

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__)


class tanh(HybridBlock):
    def __init__(self, **kwargs):
        super(tanh, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.tanh(x)

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__)
```

