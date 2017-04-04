# chainer imple. of BEGAN
paper: arXiv:1703.10717v1

This implementation lacks dataset initialization because
I couldn't get celebA dataset.

So, if you try BEGAN using this implementation,
you have to define path to celebA dataset and `chainer.datasets.ImageDataset` instance.

The code will be like bellow, instead of [L32 in `train_began.py`](https://github.com/crcrpar/chainer-BEGAN/blob/master/train_began.py#L32)

```train_began.py
image_files = os.listdir(conf['dataset'])
dataset = chainer.datasets.ImageDataset(image_files, conf['dataset'])
```

and in [L13 in `train_conf.yml`](https://github.com/crcrpar/chainer-BEGAN/blob/master/train_conf.yml#L13)

```train_conf.yaml
dataset: 'path/to/celebA_dataset'
```

Note that I assume all the image file is in `conf['dataset']` directory.
