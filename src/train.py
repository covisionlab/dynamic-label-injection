import os

import yaml
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dsets.data_factory import data_factory
from dsets.MT_dset import MT_DSET_MAP, PIXELS_PER_CLASS_DISTR
from core.label_fusion import balanced_label_fusion
from core.losses import DiceLoss, WeightedBCE, FocalLoss, ClassBalanceLoss
from core.measures import IoUMeasure
from core.parser import get_args
from core.utils import ensure_reproducibility
from models.model_factory import model_factory

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

    # log experiment
    run_name =  f"perc_{config['exp']['data_perc']}/{config['exp']['method']}/{config['exp']['model']}/{config['exp']['seed']}"
    if config['exp']['debug']:
        wandb.init(mode='disabled')
    else:
        wandb.init(project='DS_eccv24', name=run_name, config=config)

    # checkpoint path
    if config['exp']['debug']:
        checkpoint_path = config['log']['checkpoint_folder'] + f'/debug'
    else:
        checkpoint_path = config['log']['checkpoint_folder'] + f'/{run_name}' 
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # save config in checkpoint folder
    with open(checkpoint_path + '/config.yaml', 'w') as file:
        yaml.dump(config, file)

    # config hyperparameters
    threshold = config['model']['threshold']
    bsize = config['hyp']['batch_size']
    alpha = config['hyp']['alpha']
    beta = config['hyp']['beta']

    # seed
    random_generator = ensure_reproducibility(config['exp']['seed'])
    
    # model
    model = model_factory(config, device)
    
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['hyp']['lr'], weight_decay=config['hyp']['wd'])

    # data
    defects_pool, _, _, train_loader, val_loader = data_factory(config, random_generator)

    # scheduler
    if config['hyp']['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['hyp']['epochs'], eta_min=0)
    
    # losses
    dice_fn = DiceLoss()

    if config['exp']['method'] == 'wce':
        print('Using Weighted BCE Loss')
        loss_fn = WeightedBCE(PIXELS_PER_CLASS_DISTR)
    elif config['exp']['method'] == 'focal':
        print('Using Focal Loss')
        loss_fn = FocalLoss()
    elif config['exp']['method'] == 'balanced':
        print('Using Class Balance Loss')
        loss_fn = ClassBalanceLoss(PIXELS_PER_CLASS_DISTR)
    else:
        print('Using Cross Entropy Loss')
        loss_fn = nn.BCELoss()

    # train
    max_iou = 0
    for epoch in range(config['hyp']['epochs']):
        model.train()
        mu_ce, mu_dice, mu_total, ce, dice, total, iou  = 0, 0, 0, 0, 0, 0, 0
        measure = IoUMeasure()

        pbar = tqdm(train_loader, ncols=150)
        for batch_n, (x, mask, def_id) in enumerate(pbar):
            if config['exp']['debug'] and batch_n == 5:
                break

            x = x.to(device)
            mask = mask.to(device)
            
            if 'dli' in config['exp']['method']:
                x, mask, def_id = balanced_label_fusion(x, mask, def_id, defects_pool, list(MT_DSET_MAP.values()), config['exp']['poisson_prob'])
            
            # forward
            out = model(x)

            # losses
            ce_loss = loss_fn(out, mask)
            dice_loss = dice_fn(out, mask)
            total_loss = alpha*ce_loss + beta*dice_loss

            total_loss.backward()
            optimizer.step()

            # metrics
            mu_ce += ce_loss.item()
            mu_dice += dice_loss.item()
            mu_total += total_loss.item()

            ce = mu_ce / ((batch_n+1)*bsize)
            dice = mu_dice / ((batch_n+1)*bsize)
            total = mu_total / ((batch_n+1)*bsize)
            iou = measure.update(out > threshold, mask).item()

            pbar.set_description(f"[EPOCH {epoch:03} train] ce:{ce:.4f} dice:{dice:.4f} tot:{total:.4f} iou:{iou:.4f}")
            
            optimizer.zero_grad()

        if config['hyp']['scheduler'] == 'cosine':
            scheduler.step()

        # test on validation set
        if (epoch % config['exp']['log_every'] == 0):
            print(run_name , f' - current best IoU: {max_iou:.3f}')
            wandb.log({'CE/train': ce, 'Dice/train': dice, 'IoU/train': iou, 'lr': optimizer.param_groups[0]['lr']}, step=epoch)
            with torch.inference_mode(True):
                model.eval()
                mu_ce, mu_dice, mu_total, ce, dice, total, iou  = 0, 0, 0, 0, 0, 0, 0
                measure = IoUMeasure()
                pbar= tqdm(val_loader, ncols=150)
                for batch_n, (x, mask, def_id) in enumerate(pbar):
                    if config['exp']['debug'] and batch_n == 5:
                        break

                    x = x.to(device)
                    mask = mask.to(device)

                    out = model(x)

                    # losses
                    ce_loss = loss_fn(out, mask)
                    dice_loss = dice_fn(out, mask)
                    total_loss = alpha*ce_loss + beta*dice_loss

                    # metrics
                    mu_ce += ce_loss.item()
                    mu_dice += dice_loss.item()
                    mu_total += total_loss.item()

                    ce = mu_ce / ((batch_n+1)*bsize)
                    dice = mu_dice / ((batch_n+1)*bsize)
                    total = mu_total / ((batch_n+1)*bsize)
                    iou = measure.update(out > threshold, mask).item()

                    pbar.set_description(f"[EPOCH {epoch:03} val] ce:{ce:.4f} dice:{dice:.4f} tot:{total:.4f} iou:{iou:.4f}")

                wandb.log({'CE/val': ce, 'Dice/val': dice, 'IoU/val': iou}, step=epoch)

                if iou > max_iou:
                    max_iou = iou
                    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
                    print(f'-.-.-.-.-. New best model at epoch {epoch} with mean IoU: {iou:.3f} -.-.-.-.-.')
                    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
                    torch.save(model.state_dict(), checkpoint_path + f'/best.pt')

                torch.save(model.state_dict(), checkpoint_path + f'/latest.pt')

    wandb.finish()

if __name__ == '__main__':
    main()
