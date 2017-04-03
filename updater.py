
import chainer
import chainer.functions as F


class BEGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.encoder, self.decoder, self.discriminator = kwargs.pop('models')
        super(BEGANUpdater, self).__init__(*args, **kwargs)
