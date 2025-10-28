import torch
import numpy as np
import numpy.random as random
import torchvision.transforms as transforms

def generate_noise(x, device, scale):
    noise = (torch.rand_like(x) - 0.5) * 2 * scale
    noise.requires_grad_(True)
    return noise.to(device)

def get_length(length, num_block):
    rand = np.random.uniform(size=num_block)
    rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)

def shuffle_single_dim(x, dim, num_block):
    lengths = get_length(x.size(dim), num_block)
    x_strips = list(x.split(lengths, dim=dim))
    random.shuffle(x_strips)
    return x_strips

# The block shuffle implementation is based on the code:
# https://github.com/Zhijin-Ge/TransferAttack/blob/main/transferattack/input_transformation/bsr.py
def shuffle(x, num_block):
    dims = [2, 3]
    random.shuffle(dims)
    x_strips = shuffle_single_dim(x, dims[0], num_block)
    return torch.cat([torch.cat(shuffle_single_dim(x_strip, dim=dims[1], num_block=num_block), dim=dims[1]) for x_strip in x_strips], 
                     dim=dims[0])

def I_C_transformation(x, height, width):
    # attack the undefended targets
    noise_scale = 0.07
    num_block = 3
    resize_factor = random.uniform(1, 1.40)
    
    # attack the defended targets
    # noise_scale = 0.15
    # num_block = 3
    # resize_factor = random.uniform(1, 1.10)
    
    resize = transforms.Compose([transforms.Resize((int(resize_factor * height), int(resize_factor * width))),
                                 transforms.CenterCrop((height, width))])
    noise = generate_noise(x, device=torch.device("cuda:0"), scale=noise_scale)

    return resize(shuffle(x=x, num_block=num_block) + noise)
