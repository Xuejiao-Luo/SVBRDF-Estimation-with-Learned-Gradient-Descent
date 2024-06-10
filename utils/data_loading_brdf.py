import logging
import time
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import re
from PIL import Image
from torch.utils.data import Dataset
from utils.rendererMG import png2tex
from utils.rendererMG_util import gyPIL2Array
from skimage.transform import resize


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, scale: float = 1.0, train=True):
        self.images_dir = Path(images_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.train = train

        start = time.time()
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        end = time.time()
        logging.info(f'BasicDataset init time: {end - start}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        fileExt = '.jpg'
        img_brdfs_file = Path(self.images_dir, name+fileExt)

        if self.train:
            filename_img = img_brdfs_file.name
            filename_img = re.sub(r'_ligInt_\d*.png', '.png', filename_img) # obtain filename in the folder storing brdfs (NB: not the same folder of the rendered imgs)
            brdf_file = Path(img_brdfs_file.parent.parent, img_brdfs_file.parent.name[0:-9], filename_img)
            brdfs = png2tex(brdf_file)[0]  # [-1, 1]
            str_tmp = str(img_brdfs_file).replace('.', '_')
            nums = [int(s) for s in str_tmp.split('_') if s.isdigit()]
            light = nums[-1]
            light = light/10000000.0
            light = torch.from_numpy(np.array(light, dtype=float))
            light = light.unsqueeze(0).unsqueeze(1).expand_as(brdfs[0,:,:])

        img_rendered_file = Path(img_brdfs_file)
        img = Image.open(str(img_rendered_file))
        img = gyPIL2Array(img)  # [0-255] -> [0-1]
        img = img.transpose((2, 0, 1))

        if not self.train:
            img_size = img.shape[1] if img.shape[1] <= img.shape[2] else img.shape[2]
            img = img[:, int(np.floor((img.shape[1]-img_size)/2)) : int(np.floor((img.shape[1]+img_size)/2)), int(np.floor((img.shape[2]-img_size)/2)) : int(np.floor((img.shape[2]+img_size)/2)) ]
            img = resize(img, (3, 256, 256))

        return {'image': img, 'brdfs': brdfs, 'light': light,} if self.train else {'image': img, 'image_name': name}

