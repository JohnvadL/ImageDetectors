import torchvision
import torch
import torch.utils.data
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from align_faces import align_faces

transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Grayscale(),
     align_faces(),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(0.5, 0.5)
     ]
)

# Some test code to test out the transforms

# path = os.path.join(sys.path[0], "data")
# train_set = torchvision.datasets.ImageFolder(root=path, transform=transforms)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=3, shuffle=True, num_workers=2)
#
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
