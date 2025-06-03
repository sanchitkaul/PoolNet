import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class Patch(nn.Module):
    def __init__(self, patch_size=16, stride=16):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride
    
    def patchify(self, x):
        # find input shape (width, height, assume 3 channels)
        width = x.shape[-2]
        height = x.shape[-1]

        print(f"Input shape: {x.shape}, Width: {width}, Height: {height}")

        if width % self.patch_size != 0:
            width_pad = (width + self.patch_size) - (width // self.patch_size) * self.patch_size
        if height % self.patch_size != 0:
            height_pad = (height + self.patch_size) - (height // self.patch_size) * self.patch_size
        
        print(f"Padding: Width: {width_pad}, Height: {height_pad}")

        # apply padding to right and bottom
        if width_pad > 0 or height_pad > 0:
            x = nn.functional.pad(x, (0, width_pad, 0, height_pad), mode='constant', value=0)
            width += width_pad
            height += height_pad

            new_width = x.shape[-2]
            new_height = x.shape[-1]
            print(f"New shape after padding: {x.shape}, Width: {new_width}, Height: {new_height}")
    def forward(self, x):
        pass

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass



if __name__ == "__main__":
    # open image
    image_path = "/home/joseph/Projects/ECS271/Project/PoolNet/model/test-videos/test_im.jpeg"
    image = Image.open(image_path).convert("RGB")

    # convert to tensor
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

    patch_encoder = Patch(patch_size=16, stride=16)
    patches = patch_encoder.patchify(image)