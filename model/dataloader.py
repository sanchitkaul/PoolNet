import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import json
from PIL import Image
import random
from collections.abc import Iterable

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
    
        self.current_index = 0
        self.process_inorder = True # in case we want to try augmenting our data processing with out-of-order sampling
    
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
        return not all(image["used"] for image in self.images.values())
    
    def __iter__(self):
        self.reset()
        self.current_index = 0
        for key in self.process_order:

            # NOTE: add in any image processing as needed HERE
            image = self.images[key]["img"]
            label = self.label

            yield image, label, self.scene_id, key

class SceneDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.inorder = True
        self.augment = True

        self.scenes = []

        for scene_name in self.data_dir.iterdir():
            if scene_name.is_dir():
                scene_id = str(scene_name.name)
                print(f"Found scene: {scene_id}")
                scene_path = self.data_dir / scene_name

                scene = Scene(scene_id, scene_path)
                self.scenes.append(scene)

    def set_inorder(self, inorder_new):
        self.inorder = inorder_new

    def set_augment(self, augment_new):
        self.augment = augment_new

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        self.scenes[idx].set_inorder(self.inorder)
        self.scenes[idx].reset()
        return self.scenes[idx]

        # depending on settings, return in order, out of order, and with augmentation

def collate_fn(batch):
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