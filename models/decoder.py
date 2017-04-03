import chainer
import chainer.functions as F
import chainer.links as L


class Decoder(chainer.Chain):
    """ Decoder for BEGAN

    Input is embedded feature whose shape is user-defined `h`.
    This implementation uses `F.unpooling_2d` as upsampling.
    """

    def __itit__(self, n):
        self._train = True
        self.n = n
        args = {
                'ksize': 3,
                'stride': 1,
                'padding': 4,
                'initialW': chainer.initializers.HeNormal()
                }
        self._layers['fc'] = L.Linear(None, 8 * 8 * n)
        for i in range(1, 9):
            self._layers['conv_{}'.format(i)] = L.Convolution2D(n, n, **args)
        self._layers['conv_9'] = L.Convolution2D(n, 3, **args)
        super(Decoder, self).__init__(**self._layers)

    def __call__(self, h):
        return self.decode(h)

    def decode(self, h):
        fv = F.elu(self.fc(h))
        fv = F.reshape(fv, (-1, self.n, 8, 8))
        h1 = F.elu(self.conv_1(fv))
        h2 = F.elu(self.conv_2(h1))
        h2_us = F.unpooling_2d(h2, 2)
        h3 = F.elu(self.conv_3(h2_us))
        h4 = F.elu(self.conv_4(h3))
        h4_us = F.unpooling_2d(h4, 2)
        h5 = F.elu(self.conv_5(h4_us))
        h6 = F.elu(self.conv_6(h5))
        h6_us = F.unpooling_2d(h6, 2)
        h7 = F.elu(self.conv_7(h6_us))
        h8 = F.elu(self.conv_8(h7))
        out = self.conv_9(h8)
        return out
