import os
import yaml
from tqdm import tqdm

import torch
import numpy as np

from core.parser import get_args
from core.utils import ensure_reproducibility
from dsets.data_factory import data_factory
from dsets.MT_dset import MT_DSET_DEFECTS
import models.segmentation_models_pytorch as smp
from core.measures import IoUMeasure

import rootutils
rootutils.setup_root(__file__, indicator="configs", pythonpath=True)



def main():
    args = get_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config['exp'] = args.__dict__
    
    # set device
    if config['exp']['cpu']:
        device = 'cpu'
    else:
        device = 'cuda'

    threshold = config['model']['threshold']

    # Load a pretrained model
    if config['exp']['load_weights'] is None:
        raise ValueError('Please provide a path to the model weights')
    else:
        weights_files = os.listdir(config['exp']['load_weights'])
        weights_files = sorted([int(x) for x in weights_files])
        if len(weights_files) == 0:
            raise ValueError('No weights found in the provided path')
        
        iou_total_scores = []
        iou_per_class_scores = {x:[] for x in MT_DSET_DEFECTS}
        
        print(f'Method {config["exp"]["method"]} - Model {config["exp"]["model"]} - Data Perc {config["exp"]["data_perc"]}')
        
        for weights_file in weights_files:
            config['exp']['seed'] = weights_file
            random_generator = ensure_reproducibility(config['exp']['seed'])
            
            # model
            model = smp.Unet(
                encoder_name=config['exp']['model'],
                in_channels=config['model']['input_channels'],
                classes=config['model']['output_channels'],
                activation='sigmoid',
                encoder_depth=3,   # works best with small number here (3), but also with 5 works
                decoder_channels=(128, 64, 32),
                decoder_use_batchnorm = 'True'
            )
            model.to(device)
            weights_dir = config['exp']['load_weights'] + f"/{weights_file}" + "/best.pt"
            model.load_state_dict(torch.load(weights_dir, map_location=device))
        
            # data
            _, _, _, _, val_loader = data_factory(config, random_generator)

            # test loop
            with torch.inference_mode(True):
                model.eval()
                iou_total = IoUMeasure()
                iou_per_class = {x:IoUMeasure() for x in MT_DSET_DEFECTS}     
                for x, mask, def_id in tqdm(val_loader):
                    x = x.to(device)                                                                  
                    mask = mask.to(device)

                    out = model(x)
                    
                    # losses
                    _ = iou_total.update(out > threshold, mask).item()
                    _ = iou_per_class[MT_DSET_DEFECTS[def_id.item()]].update(out > threshold, mask).item()
                
                for k, v in iou_per_class.items():
                    iou_per_class_scores[k].append(v.get()[0].item())
                iou_total_scores.append(iou_total.get()[0].item())

        print('IoU per class:')
        for k,v in iou_per_class_scores.items():
            print(f'{k} = {np.mean(v):.3f} +- {np.std(v):.3f}')
        
        print(f'Total IoU = {np.mean(iou_total_scores):.3f} +- {np.std(iou_total_scores):.3f}')
        print('')


if __name__ == '__main__':
    main()
