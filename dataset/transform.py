"""

Author: Andreas Rössler
"""
from tempfile import tempdir
from torchvision import transforms
import torch

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # print("tensor type: ", type(tensor[0]), tensor[0].shape)
        # temp = tensor[0]
        # print(tensor.shape)
        tensor = (tensor-self.mean)/self.std
        # for t, m, s in zip(tensor, self.mean, self.std):
        # print("lllllllllllllllllll", t.shape, m, s)
        # print(tensor[0])
        
        # for temp in tensor:
        #     print(temp.shape)
        #     print(self.mean.shape)
            
        #     print(self.std.shape)
        # temp = tensor[0]

        # temp = temp.sub_(self.mean).div_(self.std)
        # print(tensor)
            # t.sub_(m).div_(s)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

mean = (torch.ones(1, 3, 299, 299)*0.5).cuda()
std = (torch.ones(1, 3, 299, 299)*0.5).cuda()


xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),

    # Added these transforms for attack
    'to_tensor': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'normalize': transforms.Compose([
        Normalize(mean, std)
    ]),
    
    'unnormalize' : transforms.Compose([
        UnNormalize(mean, std)
    ])
}



"""

Author: Honggu Liu
"""
mesonet_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),

    # Added these transforms for attack
    'to_tensor': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'normalize': transforms.Compose([
        Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'unnormalize' : transforms.Compose([
        UnNormalize([0.5] * 3, [0.5] * 3)
    ])
}