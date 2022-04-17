
import os
import glob
import ntpath
import numpy as np

from typing import Dict
from torch.utils.data import Dataset

from gimp_labeling_converter import generate_cmap, gimp_helper, phmError

class XCFDataset(Dataset):
    def __init__(self, 
        root_dir : str, 
        category : Dict[str, int],
        transform = None,
        target_transform = None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        if not os.path.isdir(self.root_dir):
            raise phmError('Directory path is invalid!')
        if category is None or len(category.keys()) == 0:
            raise phmError('Category should be given!')
        self.category = category
        self.transform = transform
        self.target_transform = target_transform
        self.files = glob.glob(os.path.join(root_dir,'*.xcf'))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        res = generate_cmap(file=file, helper=gimp_helper, category=self.category)
        img = np.asarray(res['original'])
        target = res['target']
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target