# chainer imple. of BEGAN
paper: arXiv:1703.10717v1

This implementation lacks handling dataset initialization because
I couldn't get celebA dataset.
So, if you try BEGAN using this implementation, you should define
celebA dataset path and `chainer.datasets.ImageDataset` instance.

The code should be like bellow:

```python
image_files = os.listdir('conf['dataset'])
dataset = chainer.datasets.ImageDataset(image_files, conf['dataset'])
```

```train_conf.yaml
...
dataset: 'path/to/celebA_dataset
```
