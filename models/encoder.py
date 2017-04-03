# coding: utf-8
import chainer
import chainer.functions as F
import chainer.links as L


class Encoder(chainer.Chain):
    """ Encoder for BEGAN

    Input image size should be (64, 64, 3).
    This implementation uses `F.average_pooling_2d` as `subsampling`
    """

    def __init__(self, n, h_dim):
        self._train = True
        args = {
                'ksize': 3,
                'stride': 1,
                'pad': 2,
                'initialW': chainer.initializers.HeNormal()
                }
        self._layers = { 
                'conv_0': L.Convolution2D(3, n, **args),
                'conv_1': L.Convolution2D(n, n, **args),
                'conv_2': L.Convolution2D(n, 2 * n, **args),
                'conv_3': L.Convolution2D(2 * n, 2 * n, **args),
                'conv_4': L.Convolution2D(2 * n, 3 * n, **args),
                'conv_5': L.Convolution2D(3 * n, 3 * n, **args),
                'conv_6': L.Convolution2D(3 * n, 4 * n, **args),
                'conv_7': L.Convolution2D(4 * n, 4 * n, **args),
                'conv_8': L.Convolution2D(4 * n, 4 * n, **args),
                'fc': L.Linear(None, h_dim)
                }
        super(Encoder, self).__init__(**self._layers)

    def __call__(self, x):
        return self.encode(x)

    def encode(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        h0 = F.elu(self.conv_0(x))
        h0_ss = F.average_pooling_2d(h0, 2)
        h1 = F.elu(self.conv_1(h0_ss))
        h2 = F.elu(self.conv_2(h1))
        h2_ss = F.average_pooling_2d(h2, 2)
        h3 = F.elu(self.conv_3(h2_ss))
        h4 = F.elu(self.conv_4(h3))
        h4_ss = F.average_pooling_2d(h4, 2)
        h5 = F.elu(self.conv_5(h4_ss))
        h6 = F.elu(self.conv_6(h5))
        h6_ss = F.average_pooling_2d(h6, 2)
        h7 = F.elu(self.conv_7(h6_ss))
        h8 = F.elu(self.conv_8(h7))
        embedded = self.fc(h8)
        return embedded

    @property
    def train(self):
        return self._train

    @property
    def change_mode(self, train):
        self._train = train


if __name__ == '__main__':
    import numpy as np
    bs = 100
    print('# batch size: {}'.format(bs))
    encoder = Encoder(16, 100)
    shape = (bs, 3, 64, 64)
    x = np.random.normal(size=shape).astype(np.float32)
    out = encoder(x)
    print('# out.shape: {}'.format(out.shape))
