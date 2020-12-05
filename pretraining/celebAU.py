import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from align_faces_with_landmarks import align_faces_with_landmarks
from torch.utils.data import Dataset, DataLoader


class CelebAU(Dataset):
    """CelebAU dataset labeled by presence of action units (AU)"""

    def __init__(self, train=False, intensity=False, transform=None):
        """
        Args:
          - label_csv: Path to the csv file with action unit labels.
          - train: training set if True, otherwise validation set
          - intensity (bool): labels are intensities (between 0 and 5) rather
                              than presence (either 0 or 1).
          - transform: transform applied to an image input
        """
        self.train = train
        if train:
            label_path = '/home/john/Documents/CSC420/data/pretraining/training/train_labels.csv'
            self.root_dir = '/home/john/Documents/CSC420/data/pretraining/training/training'
        else:
            label_path = '//home/john/Documents/CSC420/data/pretraining/validation/val_labels.csv'
            self.root_dir = '/home/john/Documents/CSC420/data/pretraining/validation/validation'
        self.au_frame = pd.read_csv(label_path, index_col=[0, 1])
        if intensity:
            self.label_cols = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
                               ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r',
                               ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r',
                               ' AU26_r', ' AU28_c', ' AU45_r']
        else:
            self.label_cols = [' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c',
                               ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c',
                               ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c',
                               ' AU26_c', ' AU28_c', ' AU45_c']

        self.landmark_cols = ['name', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y',
                              'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']

        self.intensity = intensity
        self.transform = transform

        # code to handle facial landmarks
        landmark_path = '/home/john/Documents/CSC420/data/pretraining/list_landmarks_align_celeba.csv'
        self.landmark_frame = pd.read_csv(landmark_path, index_col=[0])
        self.align = align_faces_with_landmarks()

    def __len__(self):
        return len(self.au_frame)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the image and its label if a face is
        detected. Otherwise, return None.
        """
        # Get image at idx
        image_id = self.au_frame.iloc[idx, 0]
        image_path = self.root_dir + '/' + str(image_id).zfill(6) + '.jpg'
        image = cv2.imread(image_path)
        image = Image.fromarray(image)

        # get landmarks for the specified file
        image_id = int(image_id)
        landmarks = self.landmark_frame.iloc[image_id - 1]

        landmarks = landmarks.tolist()

        # Get AU labels
        aus = self.au_frame.iloc[idx][self.label_cols]
        aus = np.array(aus, dtype=float)

        if self.transform:
            try:
                # torchvision transforms can't take multiple parameters so splitting the transforms here reference:
                # https://discuss.pytorch.org/t/t-compose-typeerror-call-takes-2-positional-arguments-but-3-were
                # -given/62529
                image = self.align(image, landmarks)
                image = self.transform(image)

            except ValueError:
                return None

        sample = {'image': image, 'labels': aus}

        return sample


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ]
)


def collate_fn(batch):
    """
  Used to process the list of samples to form a batch. Ignores images
  where no faces were detected.
  """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


val_set = CelebAU(train=False, transform=transform)
val_loader = DataLoader(val_set, batch_size=100, collate_fn=collate_fn,
                        shuffle=True, num_workers=2)


def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.show()


dataiter = iter(val_loader)
batch = dataiter.next()
imshow(torchvision.utils.make_grid(batch['image']))

start = time.time()
for i_batch, sample_batched in enumerate(val_loader):
    # print(i_batch, sample_batched['image'].size(),
    #       sample_batched['labels'].size())
    if i_batch == 1:
        break
print(f"TIME:{time.time() - start}")