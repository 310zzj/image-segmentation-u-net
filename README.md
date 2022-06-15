# Image-Segmentation-U-Net\n\nThis is a concise PyTorch implementation of U-net, as described in the paper: https://arxiv.org/abs/1505.04597\n\n![Output image 1](public/image1.png)\n\nThis project aims to provide a straightforward implementation of the model. The only dependencies are PyTorch, NumPy, and Pillow.\n\nThere are some key differences with the original paper:\n- No padding in the pooling, which simplifies dimension handling\n- No weight balancing in the softmax to address class imbalance\n\n## Example Dataset\n\nThe example dataset is from the ISBI Challenge. More details here: http://brainiac2.mit.edu/isbi_challenge/.\n\nHere are a few example outputs from the test dataset, after 300 iterations:\n\n![Output image 2](public/image2.png)\n\n![Output image 3](public/image3.png)\n\nTo use, download and place the files in the `data` directory. It should resemble this structure:\n```
data
├── test-volume.tif
├── train-labels.tif
└── train-volume.tif
```\n\n## Installation\n\nRun the following commands:\n```
pip install torch numpy pillow
mkdir model
```\n\n## Training\n\nTo train,