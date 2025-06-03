import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import json
from PIL import Image
import random
from collections.abc import Iterable
from tqdm import tqdm
import numpy as np
import cv2

class Scene(Iterable):
    def __init__(self, scene_id, scene_path):
        self.scene_id = scene_id
        self.scene_path = Path(scene_path)

        self.images = {}
        self.label = None

        for file in self.scene_path.iterdir():
            if file.suffix in ['.jpg', '.png']:
                key = int(file.name.split(".")[0].split("_")[-1])
                self.images[key] = {"img" : file, "used" : False}
            elif file.suffix == '.json':
                with open(file, 'r') as f:
                    self.label = json.load(f)

        
        self.height, self.width, _ = np.asarray(Image.open(self.images[0]["img"])).shape 
    
        self.current_index = 0
        self.process_inorder = True # in case we want to try augmenting our data processing with out-of-order sampling
        self.random_crop = False
        self.random_shift = False
        self.target_shape = (224, 224)

        self.max_length = 150 # max number of images to process

        self.reset()
    
    def get_label(self):
        if self.label is None:
            raise ValueError("Label not set for this scene.")
        return self.label
    
    def set_inorder(self, inorder_new):
        self.process_inorder = inorder_new

    def reset(self):
        for image in self.images.values():
            image["used"] = False
        
        if self.process_inorder:
            self.process_order = sorted(self.images.keys())
        else:
            self.process_order = random.sample(list(self.images.keys()), len(self.images))

    def __len__(self):
        return len(self.images)

    def has_next(self):
        # check if all images have used set to True
        return not all(image["used"] for image in self.images.values()) and not self.current_index >= self.max_length
    
    def img_aug(self, data):
        data = np.asarray(data)
        
        #center crop to square
        min_dim = min(data.shape[:2])

        left = (self.width - min_dim) // 2
        top = (self.height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        data = data[top:bottom, left:right]

        # resize to target shape
        data = cv2.resize(data, self.target_shape, interpolation=cv2.INTER_LINEAR)

        return data
    
    def get_next(self):
        if not self.has_next():
            raise StopIteration("No more images to process in this scene.")

        next_key = self.process_order[self.current_index]
        image = self.img_aug(Image.open(self.images[next_key]["img"]))
        label = self.label
        scene_id = self.scene_id
        key = next_key

        self.current_index += 1
        self.images[next_key]["used"] = True

        return image, label, scene_id, key
    
    def __iter__(self):
        self.reset()
        self.current_index = 0
        for key in self.process_order:

            # NOTE: add in any image processing as needed HERE
            image = self.img_aug(Image(self.images[key]["img"]))

            label = self.label

            yield image, label, self.scene_id, key

class SceneDataset(Dataset):
    def __init__(self, data_dir, dataloader_opts=None):
        self.data_dir = Path(data_dir)
        self.inorder = True

        if dataloader_opts == None:
            self.augment = False
            self.dataloader_opts = {}
        else:
            self.augment = True if dataloader_opts.get("augment", True) else False
            self.dataloader_opts = dataloader_opts

        self.scenes = []

        for scene_name in tqdm(self.data_dir.iterdir()):
            if scene_name.is_dir():
                scene_id = str(scene_name.name)
                scene_path = self.data_dir / scene_name

                scene = Scene(scene_id, scene_path)
                self.scenes.append([scene]) # NOTE: we need to wrap the scene in a list to allow dataloader to trick the dataloader into allowing batching (limited to 1)

    def set_inorder(self, inorder_new):
        self.inorder = inorder_new

    def set_augment(self, augment_new):
        self.augment = augment_new

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        self.scenes[idx][0].set_inorder(self.inorder)
        self.scenes[idx][0].reset()
        return self.scenes[idx]

        # depending on settings, return in order, out of order, and with augmentation

def collate_fn(batch):
    # custom collate function to prevent error for large batching
    if len(batch) > 1:
        raise ValueError("Batch size must be 1 for SceneDataset.")
    return batch[0]

if __name__ == "__main__":
    # Example usage
    data_dir = "/media/SharedStorage/redwood/output"
    dataset = SceneDataset(data_dir)
    
    dataset.set_inorder(True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # get sample scene from dataloader
    for i, scene in enumerate(dataloader):
        for sample in scene:
            image, label, scene_id, key = sample
            print(f"Scene ID: {scene_id}, Key: {key}, Image: {image}, Label: {label}")