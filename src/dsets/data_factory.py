import cv2
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from core.utils import seed_worker
from dsets.MT_dset import MTDset, DefectPool, get_data, MU_MT_DSET, STD_MT_DSET


def data_factory(config, random_generator):
    if config['data']['dset_name'] == 'MT_dset':
        perc = config['exp']['data_perc']
        defected_train_df, train_df, val_df = get_data(config, perc)

        height = config['data']['augmentation_resize_height']
        width = config['data']['augmentation_resize_width']

        t_train = A.Compose([
            A.Resize(height, width, interpolation=cv2.INTER_NEAREST), 
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=MU_MT_DSET, std=STD_MT_DSET),
            ToTensorV2()
        ])

        t_val = A.Compose([
            A.Resize(height, width, interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=MU_MT_DSET, std=STD_MT_DSET),
            ToTensorV2()
        ])

        train_dset = MTDset(train_df, transform=t_train)
        val_dset = MTDset(val_df, transform=t_val)
        defected_train_dset = DefectPool(defected_train_df, transform=t_train, perc=perc)
    else:
        raise ValueError(f'Dataset {config["data"]["dset_name"]} not found')

    train_loader = DataLoader(train_dset, batch_size=config['hyp']['batch_size'], 
                              shuffle=True, 
                              drop_last=False, 
                              generator=random_generator, 
                              num_workers=8,
                              worker_init_fn=seed_worker)
    
    val_loader = DataLoader(val_dset, 
                            batch_size=1, 
                            shuffle=False, 
                            drop_last=False,
                            num_workers=8,
                            generator=random_generator, 
                            worker_init_fn=seed_worker)
    
    return defected_train_dset, train_dset, val_dset, train_loader, val_loader
