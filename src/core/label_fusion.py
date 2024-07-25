from typing import Optional

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from pietorch import blend, blend_dst_numpy

from dsets.MT_dset import MT_DSET_MAP



def torchunique(distr, classes):
    freq = torch.zeros(len(classes))
    for c in classes:
        freq[c] = torch.sum(distr == c)  
    return freq


def dst1_1d_pytorch(x):
    c, h, w = x.shape
    transformed = torch.zeros_like(x, device=x.device)
    
    for i in range(c):
        x_i = x[i, :, :]
        N = h
        
        # Create the DST-I transformation matrix
        k = torch.arange(1, N + 1, dtype=x.dtype, device=x.device).view(N, 1)
        n = torch.arange(1, N + 1, dtype=x.dtype, device=x.device).view(1, N)
        sin_matrix = torch.sin(np.pi * k * n / (N + 1)) * torch.sqrt(torch.tensor(2.0 / (N + 1), dtype=x.dtype, device=x.device))
        
        # Perform the matrix multiplication for DST-I
        transformed[i, :, :] = sin_matrix @ x_i @ sin_matrix.T
    
    return transformed



def blend_dst_pytorch(target: torch.Tensor, source: torch.Tensor, mask: torch.Tensor, corner_coord: torch.Tensor,
                      mix_gradients: bool, channels_dim: Optional[int] = None):
    num_dims = target.ndim
    if channels_dim is not None:
        channels_dim %= num_dims  # Determine dimensions to operate on
    chosen_dimensions = [d for d in range(num_dims) if d != channels_dim]
    corner_dict = dict(zip(chosen_dimensions, corner_coord))

    result = target.clone()
    target_slices = [slice(corner_dict[i], corner_dict[i] + source.shape[i]) if i in chosen_dimensions else slice(None)
                     for i in range(num_dims)]
    target = target[tuple(target_slices)]

    # Zero edges of mask to avoid artifacts
    for d in range(len(mask.shape)):
        mask_slices = [[0, -1] if i == d else slice(mask.shape[i]) for i in range(mask.ndim)]
        mask[tuple(mask_slices)] = 0

    # Compute gradients
    grad_kernel_1 = torch.tensor([[0, -1, 1]], dtype=target.dtype, device=target.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    target_grads = [
        F.conv2d(target, grad_kernel_1.transpose(-1, -2), padding=[1, 0], groups=3),
        F.conv2d(target, grad_kernel_1, padding=[0, 1], groups=3)
    ]
    source_grads = [
        F.conv2d(source, grad_kernel_1.transpose(-1, -2), padding=[1, 0], groups=3),
        F.conv2d(source, grad_kernel_1, padding=[0, 1], groups=3)
    ]

    # Blend gradients, MIXING IS DONE AT INDIVIDUAL DIMENSION LEVEL!
    if mix_gradients:
        source_grads = [torch.where(torch.abs(t_g) >= torch.abs(s_g), t_g, s_g)
                        for t_g, s_g in zip(target_grads, source_grads)]

    if channels_dim is not None:
        mask = mask.unsqueeze(channels_dim)

    blended_grads = [t_g * (1 - mask) + s_g * mask for t_g, s_g in zip(target_grads, source_grads)]

    # Compute laplacian
    grad_kernel_2 = torch.tensor([[-1, 1, 0]], dtype=target.dtype, device=target.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    laplacian = sum([
        F.conv2d(blended_grads[0], grad_kernel_2.transpose(-1, -2), padding=[1, 0], groups=3),
        F.conv2d(blended_grads[1], grad_kernel_2, padding=[0, 1], groups=3)
    ])

    boundary_points = target.clone()
    boundary_slices = [slice(1, s - 1) if i in chosen_dimensions else slice(None) for i, s in enumerate(boundary_points.shape)]
    boundary_points[tuple(boundary_slices)] = 0

    # Compute laplacian for boundaries
    lap_kernel = torch.zeros([3 if i in chosen_dimensions else 1 for i in range(num_dims)], dtype=target.dtype, device=target.device)
    k_centre = [1 if i in chosen_dimensions else 0 for i in range(num_dims)]
    lap_kernel[tuple(k_centre)] = -2 * len(chosen_dimensions)
    for c_d in chosen_dimensions:
        for c_c in [0, 2]:
            k_pos = tuple([c_c if i == c_d else k for i, k in enumerate(k_centre)])
            lap_kernel[k_pos] = 1
    lap_kernel = lap_kernel.unsqueeze(0).repeat(3, 1, 1, 1)
    lap_kernel.to(target.device)

    boundary_points = F.conv2d(boundary_points, lap_kernel, padding=1, groups=3)

    # Subtract boundary's influence from laplacian
    mod_diff = laplacian - boundary_points
    mod_diff_slices = [slice(1, -1) if i in chosen_dimensions else slice(None) for i in range(num_dims)]
    mod_diff = mod_diff[tuple(mod_diff_slices)]

    # DST via FFT
    transformed = dst1_1d_pytorch(mod_diff)

    eigenvalues = [2 * torch.cos(torch.pi * (torch.arange(mod_diff.shape[i], dtype=target.dtype, device=target.device) + 1) / (target.shape[i] - 1)) - 2
                   for i in chosen_dimensions]

    denorm = sum([e.view([-1 if j == i else 1 for j in range(num_dims)])
                  for i, e in zip(chosen_dimensions, eigenvalues)])
    transformed /= denorm
    
    blended = dst1_1d_pytorch(transformed)

    res_indices = [slice(corner_dict[i] + 1, corner_dict[i] + 1 + blended.shape[i]) if i in chosen_dimensions else slice(None)
                   for i in range(num_dims)]
    result[tuple(res_indices)] = blended
    
    return result


def balanced_label_fusion(batch_x, batch_mask, batch_defect, defects_pool, classes, prob_is_poisson, unconsistent=True):
    ''' label fusion: transfer a defect from an image to another, using Poisson image editing
    classes DEVE avere [-1 come free, 0, 1,2,...defects]
    '''
    distr = batch_defect.clone()
    free_indices = torch.where(batch_defect == MT_DSET_MAP['MT_Free'])[0]

    for free_idx in free_indices:
        freq = torchunique(distr, classes[1:])
        distr[free_idx] = freq.argmin()

    # count how many free samples need to be used for each defected class
    quantities = torchunique(distr, classes[1:]) - torchunique(batch_defect, classes[1:])
    quantities = quantities.long()

    for defect_id, quantity in enumerate(quantities):
        defects_pool.set_defects(quantity.item(), defect_id)

        for i in range(quantity):
            # pop free index
            free_idx = free_indices[0]
            free_indices = free_indices[1:]

            defect_img, defect_mask, _ = defects_pool[i]
            free_img = batch_x[free_idx].clone()
            defect_mask = defect_mask.unsqueeze(0)
            
            # put images on the same device 
            defect_img, defect_mask, free_img = defect_img.to(batch_x.device), defect_mask.to(batch_x.device), free_img.to(batch_x.device)

            if torch.rand(1) < prob_is_poisson:
                # blend with the unconsistent poisson technique
                if unconsistent:
                    mask_new = F.max_pool2d(defect_mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
                    new_img = blend_dst_pytorch(free_img, defect_img, mask_new.squeeze(0), torch.tensor((0,0)), True, channels_dim=0)
                    
                    batch_x[free_idx] = new_img
                    batch_mask[free_idx, defect_id, :, :] = mask_new.squeeze(0)
                    batch_defect[free_idx] = defect_id

                # blend with the consistent poisson technique
                else:
                    new_img = blend(free_img, defect_img, defect_mask.squeeze(0), torch.tensor((0,0)), True, channels_dim=0)
                    batch_x[free_idx] = new_img
                    batch_mask[free_idx, defect_id, :, :] = defect_mask.squeeze(0)
                    batch_defect[free_idx] = defect_id
            else:
                # blend with the simple fusion technique
                new_img = (defect_img * defect_mask) + ((1 - defect_mask) * free_img)
                batch_x[free_idx] = new_img
                batch_mask[free_idx, defect_id, :, :] = defect_mask.squeeze(0)
                batch_defect[free_idx] = defect_id

    return batch_x, batch_mask, batch_defect
