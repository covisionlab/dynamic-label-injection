import torch
import models.segmentation_models_pytorch as smp

def model_factory(config, device):
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

    # Load a pretrained model
    if config['exp']['load_weights'] is not None:
        model.load_state_dict(torch.load(config['exp']['load_weights'], map_location=device))
        print(f'Loaded model weights from {config["exp"]["load_weights"]}')

    return model
