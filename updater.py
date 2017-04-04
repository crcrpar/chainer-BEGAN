
import chainer
import chainer.functions as F


class BEGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._lambda = kwargs.pop('Lambda')
        self._gamma = kwargs.pop('gamma')
        self._sum_loss_D = .0
        self._sum_loos_G = .0
        super(BEGANUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        if self.is_new_epoch:
            self.clear_loss()

        opt_gen = self.get_optimizer('opt_gen')
        opt_dis = self.get_optimizer('opt_dis')

        batch = self.get_iterator('main').next()
        x = chainer.Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x.data)

        gen = self.gen
        dis = self.dis
        batchsize = len(batch)

        # calculate loss for real data
        out_real = dis(x)
        reconstruction_loss_real = F.mean_absolute_error(out_real, x)

        # calculate loss for fake data
        z = chainer.Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        out_fake = dis(x_fake)
        reconstruction_loss_fake = F.mean_absolute_error(out_fake, x_fake)

        # calculate loss for each model
        loss_D = reconstruction_loss_real - self.k_t * reconstruction_loss_fake
        loss_G = reconstruction_loss_fake

        # update
        gen.clear_grads()
        loss_G.backward()
        opt_gen.update()

        dis.clear_grads()
        loss_D.backward()
        opt_dis.update()

        # calculate convergence measure
        m_global = reconstruction_loss_real + \
            F.mean_absolute_error(
                self._gamma * reconstruction_loss_real, reconstruction_loss_fake)

        chainer.report({
            'gen/loss': loss_G,
            'dis/loss': loss_D,
            'convergence': m_global})

        self._sum_loss_D += loss_D.data
        self._sum_loos_G += loss_G.data

    def gamma(self, fake_recon, real_recon):
        self._gamma = fake_recon.data / real_recon.data

    def clear_loss(self):
        if self.epoch > 0:
            self.k_t += self._lambda * (self.gamma *
                                        self._sum_loss_D - self._sum_loss_D)
        else:
            self.k_t = 0.0
        self._sum_loss_D = .0
        self._sum_loos_G = .0
