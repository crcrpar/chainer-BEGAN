import chainer
import chainer.functions as F
import chainer.links as L


class AutoEncoder(chainer.Chain):
    """ Discriminator for BEGAN

    This autoencoder consists of `encoder.Encoder` and `decoder.Decoder`
    """

    def __init__(self, n, h_dim):
        self._train = True
        self.n = n
        self.h_dim = h_dim
        kwargs = {
                'ksize': 3,
                'stride': 1,
                'pad': 1,
                'initialW': chainer.initializeers.HeNormal()
                }
        self._layers = {
                'e_conv_0': L.Convolution2D(3, n, **kwargs),
                'e_conv_1': L.Convolution2D(n, n, **kwargs),
                'e_conv_2': L.Convolution2D(n, 2 * n, **kwargs),
                'e_conv_3': L.Convolution2D(2 * n, 2 * n, **kwargs),
                'e_conv_4': L.Convolution2D(2 * n, 3 * n, **kwargs),
                'e_conv_5': L.Convolution2D(3 * n, 3 * n, **kwargs),
                'e_conv_6': L.Convolution2D(3 * n, 4 * n, **kwargs),
                'e_conv_7': L.Convolution2D(4 * n, 4 * n, **kwargs),
                'e_conv_8': L.Convolution2D(4 * n, 4 * n, **kwargs),
                'e_fc': L.Linear(None, h_dim),
                'd_fc': L.Linear(None, 8 * 8 * n)
                }
        for i in range(1, 9):
            self._layers['d_conv_{}'.format(i)] = L.Convolution2D(n, n, **kwargs)
        self._layers['d_conv_9'] = L.Convolution2D(**self._layers)
        super(AutoEncoder, self).__init__(**self._layers)

    def __call__(self, x):
        return self.discriminate(x)

    def discriminate(self, x):
        out = self.decode(self.encode(x))
        return out

    def encode(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x, volatile=not self._train)
        h0 = F.elu(self.e_conv_0(x))
        h0_ss = F.average_pooling_2d(h0, 2)
        h1 = F.elu(self.e_conv_1(h0_ss))
        h2 = F.elu(self.e_conv_2(h1))
        h2_ss = F.average_pooling_2d(h2, 2)
        h3 = F.elu(self.e_conv_3(h2_ss))
        h4 = F.elu(self.e_conv_4(h3))
        h4_ss = F.average_pooling_2d(h4, 2)
        h5 = F.elu(self.e_conv_5(h4_ss))
        h6 = F.elu(self.e_conv_6(h5))
        h6_ss = F.average_pooling_2d(h6, 2)
        h7 = F.elu(self.e_conv_7(h6_ss))
        h8 = F.elu(self.e_conv_8(h7))
        enc_latent = self.e_fc(h8)
        return enc_latent

    def decode(self, h):
        outsize_set = ((16, 16), (32, 32), (64, 64))
        n = self.n
        fv = F.elu(self.d_fc(h))
        fv = F.reshape(fv, (-1, n, 8, 8))
        h1 = F.elu(self.d_conv_1(fv))
        h2 = F.elu(self.d_conv_2(h1))
        h2_us = F.unpooling_2d(h2, 2, outsize=outsize_set[0])
        h3 = F.elu(self.d_conv_3(h2_us))
        h4 = F.elu(self.d_conv_4(h3))
        h4_us = F.unpooling_2d(h4, 2, outsize=outsize_set[1])
        h5 = F.elu(self.d_conv_5(h4_us))
        h6 = F.elu(self.d_conv_6(h5))
        h6_us = F.unpooling_2d(h6, 2, outsize=outsize_set[2])
        h7 = F.elu(self.d_conv_7(h6_us))
        h8 = F.elu(self.d_conv_8(h7))
        out = self.d_conv_9(h8)
        return out

    @property
    def train(self):
        return self._train
