import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pdb import set_trace as bp  


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale       
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.half_len = torch.div(len(self.ids), 2, rounding_mode='floor')
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        # return len(self.ids)
        return torch.div(len(self.ids), 2, rounding_mode='floor')

    @classmethod
    def preprocess(cls, pil_img, scale):
        # w, h = pil_img.size
        w = 256*5 # 288 
        h = 256 # 288 
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray.transpose((2, 0, 1))  # dim: [RGB, W, H]

        img = img_ndarray[:,:,0:int(newW/5)]
        # img = img/255
        
        normals = img_ndarray[0:2,:,int(newW/5):2*int(newW/5)]
        diffuse = img_ndarray[:,:,2*int(newW/5):3*int(newW/5)]
        roughness = img_ndarray[0,:,3*int(newW/5):4*int(newW/5)]
        specular = img_ndarray[:,:,4*int(newW/5):5*int(newW/5)]
        brdfs = {'normals':normals,'diffuse':diffuse, 'roughness':roughness, 'specular':specular}
        
        return img, brdfs

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        return self.ids[2*idx], idx
        # idx = self.half_len + idx
        # return self.ids[idx], idx



class BasicDataset_prediction(Dataset):
    def __init__(self, images_dir: str):
        self.images_dir = Path(images_dir)
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.') and 'normal' in file]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], idx


class BasicDataset_maps_highlightaware(Dataset):
    def __init__(self, images_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)
        # return torch.div(len(self.ids), 2, rounding_mode='floor')

    @classmethod
    def preprocess(cls, pil_img, scale):
        # w, h = pil_img.size
        w = 256 * 4  # 288
        h = 256  # 288
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray.transpose((2, 0, 1))  # dim: [RGB, W, H]

        # img = img_ndarray[:,:,0:int(newW/5)]
        #
        # normals = img_ndarray[:,:,int(newW/5):2*int(newW/5)]
        # diffuse = img_ndarray[:,:,2*int(newW/5):3*int(newW/5)]
        # roughness = img_ndarray[:,:,3*int(newW/5):4*int(newW/5)]
        # specular = img_ndarray[:,:,4*int(newW/5):5*int(newW/5)]

        diffuse = img_ndarray[:, :, 0:int(newW / 4)]
        specular = img_ndarray[:, :, int(newW / 4):2 * int(newW / 4)]
        roughness = img_ndarray[:, :, 2 * int(newW / 4):3 * int(newW / 4)]
        normals = img_ndarray[:, :, 3 * int(newW / 4):4 * int(newW / 4)]
        brdfs = {'normals': normals, 'diffuse': diffuse, 'roughness': roughness, 'specular': specular}

        # return img, brdfs
        return brdfs

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
       return self.ids[idx], idx
