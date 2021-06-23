import os
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
import h5py
from torchvision.transforms import functional
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

class ODOC(Dataset):
    """ ODOC Dataset """
    def __init__(self, base_dir=None, split='train', transform =None):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        print(os.getcwd())
        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split == 'train':
            self.is_train=True
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            self.is_train = False
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]

        self.transform = transforms.Compose([
            
            transforms.ToTensor()
        ])
     

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
       


    def _random_rotate_and_flip(self, image, label_cup, label_disc, con_cup, con_disc,
                               con_gau_cup, con_gau_disc):
        if random.random() < 0.3:
            image = TF.hflip(image)
            label_cup = TF.hflip(label_cup)
            label_disc = TF.hflip(label_disc)
            con_cup = TF.hflip(con_cup)
            con_disc = TF.hflip(con_disc)
            con_gau_cup = TF.hflip(con_gau_cup)
            con_gau_disc = TF.hflip(con_gau_disc)


        if random.random() < 0.3:
            angel = transforms.RandomRotation.get_params([-30, 30])
            image = TF.rotate(image, angel)
            label_cup = TF.rotate(label_cup, angel)
            label_disc = TF.rotate(label_disc, angel)
            con_cup = TF.rotate(con_cup, angel)
            con_disc = TF.rotate(con_disc, angel)
            con_gau_cup = TF.rotate(con_gau_cup, angel)
            con_gau_disc = TF.rotate(con_gau_disc, angel)

        if random.random() < 0.3:
            image = TF.to_grayscale(image, num_output_channels=3)

        return image, label_cup, label_disc, con_cup, con_disc, con_gau_cup, con_gau_disc




    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + '/h5py_all' + '/'+image_name, 'r')
        image = Image.fromarray((h5f['img'][:] * 255).astype(np.uint8))
        label_cup = Image.fromarray(h5f['mask'][:, :, 0])
        label_disc = Image.fromarray(h5f['mask'][:, :, 1])
        con_cup = Image.fromarray(h5f['con'][:, :, 0])
        con_disc = Image.fromarray(h5f['con'][:, :, 1])
        con_gau_cup = Image.fromarray(h5f['con_gau'][:, :, 0])
        con_gau_disc = Image.fromarray(h5f['con_gau'][:, :, 1])
       
        if self.is_train:
            image, label_cup, label_disc, con_cup, con_disc, con_gau_cup, con_gau_disc= \
                self._random_rotate_and_flip(image, label_cup, label_disc, con_cup, con_disc, con_gau_cup, con_gau_disc)

            image = self.transform(image)

            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            label_cup = self.transform(label_cup)
            label_disc = self.transform(label_disc)
            label = torch.cat((label_cup, label_disc), 0)

            con_cup = self.transform(con_cup)
            con_disc = self.transform(con_disc)
            con = torch.cat((con_cup, con_disc), 0)

            con_gau_cup = self.transform(con_gau_cup)
            con_gau_disc = self.transform(con_gau_disc)
            con_gau = torch.cat((con_gau_cup, con_gau_disc), 0)


            sample = {'img': image, 'mask': label, 'con': con, 'con_gau': con_gau}

            return sample
        else:
            image = self.test_transform(h5f['img'][:])
            label = functional.to_tensor(h5f['mask'][:])
            con = functional.to_tensor(h5f['con'][:])
            con_gau = functional.to_tensor(h5f['con_gau'][:])
            keypoints = functional.to_tensor(h5f['keypoints'][:])
            sample = {'img': image, 'mask': label, 'con': con, 'con_gau': con_gau}
            return sample, image_name




