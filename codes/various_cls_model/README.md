# Various Image Classification Model on CIFAR-10

This is a repo to exercise and make me more familiar to use pytorch framework.

## Supported Module (See model_utils.py)

- Conventional Convolutional Block
- Depth-Wise Convolutional Block
- Channel-Shuffle Block
- Group Convolutional Block
- Residual Block
- Focus Module (which is proposed in YOLOv4 and YOLOv5, a downsample version of "Pixel Shuffle Operation")
- SE Block
- Residual Shuffle Block (residual + shuffle)
- Residual SE Block (residual + SE)

## Supported Backbone

- PCA + RBF-Kernel SVM

- Tiny Hierarchical CNN (similar to VGG)

- Tiny ResNet
- Tiny SENet (Tiny ResNet with SE module)
- Tiny ShuffleNet(Depth-Wise Cons & Channel-Shuffle)

## Performance

You can get it in report.pdf file.

