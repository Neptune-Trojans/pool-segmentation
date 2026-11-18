"""The pool segmentation network: finds the pool's water in a frame (docs/model.md).

    network.py   the U-Net (ImageNet-pretrained ResNet encoder)
    data.py      training frames and masks from the labeling dataset, augmentation, the split
    train.py     training and validation (python -m pool_segmentation.model train)
    runs.py      the training runs' folders: find_model()
    predict.py   a trained model: class per pixel of a frame, the water's outline

All but runs.py import torch: import them inside the function that needs a model, so tools that don't (the labeling
commands, the Labels page) start fast.
"""
