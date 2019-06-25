from __future__ import print_function
import io
import os
import sys
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import dlib
import numpy as np
from skimage.morphology import convex_hull_image

detector = dlib.get_frontal_face_detector()
kpt_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def boundingRect(img):
    h, w = img.shape[:2]
    for x0 in range(w):
        if np.count_nonzero(img[:, x0, ...]) > 0:
            break
    for x1 in range(w-1, -1, -1):
        if np.count_nonzero(img[:, x1, ...]) > 0:
            break
    for y0 in range(h):
        if np.count_nonzero(img[y0, ...]) > 0:
            break
    for y1 in range(h-1, -1, -1):
        if np.count_nonzero(img[y1, ...]) > 0:
            break
    #print('x: %d: %d\ny: %d: %d'%(x0, x1, y0, y1))
    assert x0 <= x1 and y0 <= y1
    return x0, x1, y0, y1

def convexFace(img_bgr):
  faces = detector(img_bgr, 1)
  if len(faces) == 0:
    return np.zeros_like(img_bgr)
  kpt = kpt_predictor(img_bgr, faces[0])
  kpt_mask = np.zeros_like(img_bgr[..., 0], dtype=bool)
  for i in range(68):
    x, y = kpt.part(i).x, kpt.part(i).y
    kpt_mask[min(max(y, 0), kpt_mask.shape[0]-1), min(max(x, 0), kpt_mask.shape[1]-1)] = 1
  chull = convex_hull_image(kpt_mask)
  x0, x1, y0, y1 = boundingRect(chull)
  return (img_bgr*chull[..., np.newaxis])[y0: y1, x0: x1, ...]

class ListDataset(data.Dataset):
    def __init__(self, root, list_file, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.transform = transform

        self.fname = []
        self.fiiqa = []

        with io.open(list_file, encoding='gbk') as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            sp = line.strip().split()
            self.fname.append(sp[0])
            self.fiiqa.append(int(sp[1]))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          fiiqa: (float) fiiqa.
        '''
        # Load image and bbox locations.
        fname = self.fname[idx]
        fiiqa = self.fiiqa[idx]

        img = Image.open(os.path.join(self.root, fname)).convert('RGB')
        img = convexFace(np.array(img)[..., ::-1])[..., ::-1]
        img = self.transform(Image.fromarray(img))
        return img, fiiqa

    def __len__(self):
        return self.num_imgs


def test():
    import torchvision
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    dataset = ListDataset(root='./data/validationset/val-faces/', list_file='./data/validationset/val-faces/new_4people_val_standard.txt', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    for img, fiiqa in dataloader:
        print(img.size())
        print(fiiqa.size())

