#!/usr/bin/env python
# coding: utf-8
import os
import json
import yaml
import chainer
from chainer import training
from chainer.training import extensions

import models
from updater import BEGANUpdater
import utils


def main():
    with open('train_conf.yml', 'r') as f:
        conf = yaml.load(f)
    print('# training config')
    print(json.dumps(conf, indent=2))

    generator = models.Decoder(conf['n'])
    discriminator = models.AutoEncoder(conf['n'], conf['h'])
    if conf['gpu'] >= 0:
        chainer.cuda.get_device(conf['gpu']).use()
        generator.to_gpu()
        discriminator.to_gpu()
    opt_gen = chainer.optimizers.Adam()
    opt_gen.setup(generator)
    opt_dis = chainer.optimizers.Adam()
    opt_dis.setup(discriminator)

    dataset = utils.get_dataset(conf['dataset'])
    if conf['parallel']:
        train_iter = chainer.iterators.MultiprocessIterator(dataset, conf['bastchsize'])
    else:
        train_iter = chainer.iterators.SerialIterator(dataset, conf['bastchsize'])
    updater = BEGANUpdater(
        models=(generator, discriminator),
        Lambda=conf['lambda'],
        gamma=conf['gamma'],
        iterator=train_iter,
        optimizers={
            'opt_gen': opt_gen, 'opt_dis': opt_dis},
        device=conf['gpu'])
    trainer = training.Trainer(updater, (conf['epoch'], 'epoch'), out=conf['out'])
    snapshot_interval = (conf['snapshot_interval'], 'iteration')
    display_interval = (conf['display_interval'], 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        generator, 'generator_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        discriminator, 'discriminator_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'dis/loss', 'gen/loss', 'convergence', ]),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    if conf['resume']:
        chainer.serializers.load_npz(conf['resume'], trainer)

    trainer.run()


if __name__ == '__main__':
    main()
