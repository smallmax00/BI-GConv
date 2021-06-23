import os

import numpy as np

from torch.utils.data import Dataset
import h5py
from torchvision.transforms import functional
import torchvision.transforms.functional as TF
import random

import torchvision.transforms as transforms
from PIL import Image

class polyp(Dataset):
    """ polyp SKIN Dataset """
    def __init__(self, base_dir=None, split='train', transform =None):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        print(os.getcwd())
        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split == 'train':
            self.is_train = True
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
        

    def _random_rotate_and_flip_grayscale(self, image, label, con, con_gau):
        if random.random() < 0.3:
            image = TF.hflip(image)
            label = TF.hflip(label)

            con = TF.hflip(con)

            con_gau = TF.hflip(con_gau)

            


        if random.random() < 0.3:
            angel = transforms.RandomRotation.get_params([-30, 30])
            image = TF.rotate(image, angel)
            label = TF.rotate(label, angel)

            con = TF.rotate(con, angel)

            con_gau = TF.rotate(con_gau, angel)

          

        if random.random() < 0.3:
            image = TF.to_grayscale(image, num_output_channels=3)

        return image, label, con, con_gau


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + '/h5py_all' + '/'+image_name, 'r')
        image = TF.resize(img=Image.fromarray((h5f['img'][:] * 255).astype(np.uint8)), size=(256, 256), interpolation=0)
        label = TF.resize(img=Image.fromarray(h5f['mask'][:]), size=(256, 256), interpolation=0)
        con = TF.resize(img=Image.fromarray(h5f['con'][:]), size=(256, 256), interpolation=0)
        con_gau = TF.resize(img=Image.fromarray(h5f['con_gau'][:]), size=(256, 256), interpolation=0)
    


        if self.is_train:

            image, label, con, con_gau = self._random_rotate_and_flip_grayscale(image, label, con, con_gau)

            image = self.transform(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            label = self.transform(label)
            con = self.transform(con)
            con_gau = self.transform(con_gau)
            
            sample = {'img': image, 'mask': label, 'con': con, 'con_gau': con_gau}
            return sample

        else:
            image = self.test_transform(image)
            label = functional.to_tensor(label)
            con = functional.to_tensor(con)
            con_gau = functional.to_tensor(con_gau)
            sample = {'img': image, 'mask': label, 'con': con, 'con_gau': con_gau}
            return sample, image_name



