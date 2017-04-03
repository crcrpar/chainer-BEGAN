#!/usr/bin/env python
# coding: utf-8
import os
import json
import yaml
import numpy as np
import chainer
from chainer import training
from chainer.training import extensions

from models import Encoder
from models import Decoder
from updater import BEGANUpdater
import utils


def main():
    with open('train_conf.yml', 'r') as f:
        conf = yaml.load(f)
    print('# training config')
    print(json.dumps(conf, indent=2))

    enc = Encoder(conf['n'], conf['h'])
    dec = Decoder(conf['n'])
    if conf['gpu'] >= 0:
        chainer.cuda.get_device(conf['gpu']).use()
        enc.to_gpu()
        dec.to_gpu()
    opt_enc = chainer.optimizers.Adam()
    opt_enc.setup(enc)
    opt_dec = chainer.optimizers.Adam()
    opt_dec.setup(dec)

    dataset = utils.get_dataset(conf['dataset'])
    if conf['parallel']:
        train_iter = chainer.iterators.MultiprocessIterator(dataset, conf['bastchsize'])
    else:
        train_iter = chainer.iterators.SerialIterator(dataset, conf['bastchsize'])
    updater = BEGANUpdater(
        models=(enc, dec),
        iterator=train_iter,
        optimizers={
            'enc': opt_enc, 'gen': opt_gen},
        device=conf['gpu'])
    trainer = training.Trainer(updater, (conf['epoch'], 'epoch'), out=conf['out'])
    snapshot_interval = (conf['snapshot_interval'], 'iteration')
    display_interval = (conf['display_interval'], 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'decoder_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'encoder_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'dec/loss', 'enc/loss',]),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    if conf['resume']:
        chainer.serializers.load_npz(conf['resume'], trainer)

    trainer.run()


if __name__ == '__main__':
    main()
