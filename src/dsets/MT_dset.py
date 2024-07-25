import torch
import cv2
from torch.utils.data import Dataset
import glob
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

MT_DSET_MAP = {  
    'MT_Free': -1,
    'MT_Blowhole': 0, 
    'MT_Break': 1, 
    'MT_Crack': 2, 
    'MT_Fray': 3, 
    'MT_Uneven': 4,
}
MT_DSET_DEFECTS = ['Blowhole', 'Break', 'Crack', 'Fray', 'Uneven']
PIXELS_PER_CLASS_DISTR = [0.2490, 0.2294, 0.2483, 0.2005, 0.0729]
MU_MT_DSET = (0.4319174835105347, 0.4319174835105347, 0.4319174835105347)
STD_MT_DSET = (0.11959978240241884, 0.11959978240241884, 0.11959978240241884)
HEIGHT_RESIZE, WIDTH_RESIZE = 256, 256

TRANS_DICTIONARY = [
    A.Compose([
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5, rotate_limit=(-10, 10), shift_limit=(-0.0625, 0.0625)),
        A.Resize(HEIGHT_RESIZE, WIDTH_RESIZE, interpolation=cv2.INTER_NEAREST),
        ToTensorV2()
    ]),
    
    A.Compose([
        A.Flip(p=0.5),
        A.Resize(HEIGHT_RESIZE, WIDTH_RESIZE, interpolation=cv2.INTER_NEAREST),
        ToTensorV2()
    ]),

    A.Compose([
        A.ShiftScaleRotate(p=0.5, rotate_limit=(-5, 5), shift_limit=(-0.0625, 0.0625)),
        A.Resize(HEIGHT_RESIZE, WIDTH_RESIZE, interpolation=cv2.INTER_NEAREST),
        ToTensorV2()

    ]),

    A.Compose([
        A.Flip(p=0.5),
        A.Resize(HEIGHT_RESIZE, WIDTH_RESIZE, interpolation=cv2.INTER_NEAREST),
        ToTensorV2()

    ]),

    A.Compose([
        A.Flip(p=0.5),
        A.Resize(HEIGHT_RESIZE, WIDTH_RESIZE, interpolation=cv2.INTER_NEAREST),
        ToTensorV2()
        ]),
]


def parse_data(metal_dset_data_path, mt_dset_csv):
    img_fpaths = glob.glob(metal_dset_data_path + '/**/*.jpg', recursive=True)
    mask_fpaths = [f.replace('.jpg', '.png') for f in img_fpaths]
    csv_content = "img_fpath,mask_fpath,defect_id\n"
    for img_fpath, mask_fpath in zip(img_fpaths, mask_fpaths):
        defect_id = MT_DSET_MAP[img_fpath.split('/')[-3]]
        csv_content += f"{img_fpath},{mask_fpath},{defect_id}\n"
    
    with open(mt_dset_csv, 'w') as f:
        f.write(csv_content)

    df = pd.read_csv(mt_dset_csv)
    print(df['defect_id'].value_counts())
    print(df.head())


def get_data(config, perc):
    train_data = pd.read_csv(f"{config['data']['folder']}/{config['data']['dset_name']}/splits/{config['exp']['seed']}/train_data.csv")
    free_examples = train_data[train_data['defect_id'] == -1]
    val_data = pd.read_csv(f"{config['data']['folder']}/{config['data']['dset_name']}/splits/{config['exp']['seed']}/val_data.csv")
    
    new_train_data = []
    for defect_id in MT_DSET_MAP.values():
        if defect_id == -1: # free defect id
            continue
        tr = train_data[train_data['defect_id'] == defect_id]
        
        if perc < 1:
            # take only the percentage of the training set
            tr = tr.sample(n=int(tr.shape[0] * perc))
        new_train_data.append(tr)

    train_defected_data = pd.concat(new_train_data)

    if perc < 1:
        # take only perc of free examples
        free_examples = free_examples.sample(n=int(free_examples.shape[0] * perc))
    train_data = pd.concat([train_defected_data, free_examples])

    return train_defected_data, train_data, val_data


class MTDset(Dataset):
    def __init__(self, df, transform):
        self.transform = transform
        self.multiclass = True
        self.df = df
        self.mu = torch.tensor(MU_MT_DSET, requires_grad=False).view(3,1,1)
        self.std = torch.tensor(STD_MT_DSET, requires_grad=False).view(3,1,1)

    def __getitem__(self, idx):
        img_fpath = self.df.iloc[idx]['img_fpath']
        mask_fpath = self.df.iloc[idx]['mask_fpath']
        defect_id = self.df.iloc[idx]['defect_id']

        img = cv2.cvtColor(cv2.imread(img_fpath), cv2.COLOR_BGR2RGB)
        defect_mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)

        if self.multiclass:
            mask = np.zeros((5, defect_mask.shape[0], defect_mask.shape[1]))
            mask[defect_id, :, :] = (defect_mask > 0).astype(np.float32)
            transformed = self.transform(image=img, masks=[m for m in mask])
            img = transformed['image']
            mask = torch.stack(transformed['masks'], dim=0).to(torch.float32)
        else:
            transformed = self.transform(image=img, mask=(defect_mask > 0).astype(np.float32))
            img = transformed['image']
            mask = transformed['mask'].unsqueeze(0)

        return img, mask, defect_id

    def __len__(self):
        return len(self.df)
    
    def unnorm(self, img):
        """ Remove normalization for visualization purposes """        
        return img * self.std.to(img.device) + self.mu.to(img.device)


class DefectPool(Dataset):
    def __init__(self, df, transform, perc):
        self.transform = transform
        self.multiclass = True
        self.df = df
        self.current_df = None
        self.mu = torch.tensor(MU_MT_DSET, requires_grad=False).view(3,1,1)
        self.std = torch.tensor(STD_MT_DSET, requires_grad=False).view(3,1,1)
        self.replace = perc < 1.0
        
    def __getitem__(self, idx):
        img_fpath = self.current_df.iloc[idx]['img_fpath']
        mask_fpath = self.current_df.iloc[idx]['mask_fpath']
        defect_id = self.current_df.iloc[idx]['defect_id']
        img = cv2.cvtColor(cv2.imread(img_fpath), cv2.COLOR_BGR2RGB)
        defect_mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)

        if self.multiclass:
            mask = np.zeros((5, defect_mask.shape[0], defect_mask.shape[1]))
            mask[defect_id, :, :] = (defect_mask > 0).astype(np.float32)
            transformed = self.transform(image=img, masks=[m for m in mask])
            img = transformed['image']
            mask = torch.stack(transformed['masks'], dim=0).to(torch.float32)  
        else:   
            transformed = self.transform(image=img, mask=(defect_mask > 0).astype(np.float32))
            img = transformed['image']
            mask = transformed['mask'].unsqueeze(0)

        defect_mask = mask.sum(dim=0)
        defect = img/255
        defect = (defect - self.mu) / self.std

        return defect, defect_mask, defect_id
    
    def set_defects(self, n_examples, class_id):
        self.current_df = self.df[self.df['defect_id'] == class_id]
        self.current_df = self.current_df.sample(n=n_examples, replace=self.replace)
        self.transform = TRANS_DICTIONARY[class_id]

    def __len__(self):
        return len(self.df)
