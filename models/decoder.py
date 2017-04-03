import chainer
import chainer.functions as F
import chainer.links as L


class Decoder(chainer.Chain):
    """ Decoder for BEGAN

    Input is embedded feature whose shape is user-defined `h`.
    This implementation uses `F.unpooling_2d` as upsampling.
    """

    def __init__(self, n):
        self._train = True
        self.n = n
        args = {
                'ksize': 3,
                'stride': 1,
                'pad': 1,
                'initialW': chainer.initializers.HeNormal()
                }
        self._layers = dict()
        self._layers['fc'] = L.Linear(None, 8 * 8 * n)
        for i in range(1, 9):
            self._layers['conv_{}'.format(i)] = L.Convolution2D(self.n, self.n, **args)
        self._layers['conv_9'] = L.Convolution2D(self.n, 3, **args)
        super(Decoder, self).__init__(**self._layers)

    def __call__(self, h):
        return self.decode(h)

    def decode(self, h):
        n = self.n
        fv = F.elu(self.fc(h))
        fv = F.reshape(fv, (-1, n, 8, 8))
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

    def validate(self, h):
        n = self.n
        fv = F.elu(self.fc(h))
        fv = F.reshape(fv, (-1, n, 8, 8))
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
        print(fv.shape, h1.shape, h2.shape, h2_us.shape, h3.shape,
        h4.shape, h4_us.shape, h5.shape, h6.shape, h6_us.shape, h7.shape, h8.shape,
        out.shape)


if __name__ == '__main__':
    import six
    import numpy as np
    n = 3
    bs = 10
    x = np.random.normal(size=(bs, n)).astype(np.float32)
    decoder = Decoder(n)
    decoder.validate(x)
    out = decoder(x)
    print('out.shape: {}'.format(out.shape))
