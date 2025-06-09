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
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from processredwood import camera, point2D, point3D, angle_between_cameras, cam_center_from_qvec
from collections import OrderedDict
import math
import matplotlib.pyplot as plt

class Scene:
    def __init__(self, scene_id, scene_path):
        self.scene_id = scene_id
        self.scene_path = Path(scene_path)
        self.image_path = self.scene_path.parent.parent.parent
        self.camera_path = self.scene_path / "cameras.json"

        self.cameras = {}
        self.points2d = {}
        self.points3d = {}
        self.label = None
        # print(self.scene_id, self.scene_path, self.image_path)

        if self.camera_path.exists():
            with open(self.camera_path, 'r') as f:
                self.camera_info = json.load(f)
                for cam, data in self.camera_info.items():
                    image_id = data["image_id"]
                    qvec = data["qvec"]
                    tvec = data["tvec"]
                    cam_id = data["cam_id"]
                    fname = data["fname"]

                    cam = camera(image_id, qvec, tvec, cam_id, fname)
                    self.cameras[image_id] = cam
                    for pt_data in data["points"]:
                        if pt_data["point_idx"] not in self.points2d:
                            self.points2d[pt_data["point_idx"]] = point2D(pt_data["point_idx"], pt_data["x"], pt_data["y"])
                            self.cameras[image_id].add_point(self.points2d[pt_data["point_idx"]])
                        else:
                            self.cameras[image_id].add_point(self.points2d[pt_data["point_idx"]])
                    for point, pt_3D_data in data["points3d"].items():
                        if point not in self.points3d.keys():
                            self.points3d[point] = point3D(pt_3D_data["pt"]["point3D_id"], pt_3D_data["pt"]["xyz"], pt_3D_data["pt"]["rgb"], pt_3D_data["pt"]["error"])
                            self.cameras[image_id].add_point3D(self.points3d[point], self.points3d[point], pt_3D_data["point2d_idx"])

                        else:
                            self.cameras[image_id].add_point3D(self.points3d[point], self.points3d[point], pt_3D_data["point2d_idx"])

        img_files = [(cam.fname, cam.image_id) for cam in self.cameras.values()]
        sorted_img_files = sorted(img_files, key=lambda x: int(x[0].split(".")[0].split("_")[-1]))
        self.sequential_image = OrderedDict()

        for img_file, image_id in sorted_img_files:
            self.sequential_image[image_id] = {"filename" : img_file, "img" : self.cameras[image_id]}

        # print(self.image_path, img_files[0][0])
        self.height, self.width, _ = np.asarray(Image.open(self.image_path / img_files[0][0])).shape 
    
        self.current_index = 0
        self.process_inorder = True # in case we want to try augmenting our data processing with out-of-order sampling
        self.random_crop = False
        self.random_shift = False
        self.target_shape = (320, 240)

        self.max_length = 120 # max number of images to process
        self.max_length = 20
        self.iter_rate = 5

        # self.reset()

    def get_camera_spread(self):
        centers = []
        for cam in self.cameras.values():
            if not isinstance(cam, camera):
                raise TypeError("cam must be an instance of the camera class.")
            else:
                camera_center = cam_center_from_qvec(cam.qvec, cam.tvec)
                centers.append(camera_center)
        # calculate standard deviation of all x, y, z in camera centers
        centers = np.array(centers)
        x_std = np.std(centers[:, 0])
        y_std = np.std(centers[:, 1])
        z_std = np.std(centers[:, 2])

        return (x_std, y_std, z_std)
    
    def get_random_cam(self):
        return random.choice(list(self.cameras.values()))

    def get_pair(self, image_id):
        key_list = list(self.sequential_image.keys())
        index = key_list.index(image_id)

        sequential_distance = random.randint(1, 4)
        
        if index < len(key_list) - sequential_distance:
            next_image_id = key_list[index + sequential_distance]
        elif index > sequential_distance:
            next_image_id = key_list[index - sequential_distance]
        else:
            print(f"error, unable to find adjacent image for {image_id} in {self.sequential_image.keys()}")
        
        image1 = Image.open(self.image_path / self.sequential_image[image_id]["filename"])
        image1_data = self.img_aug(image1)
        cam1_data = self.cameras[image_id]

        image2 = Image.open(self.image_path / self.sequential_image[next_image_id]["filename"])
        image2_data = self.img_aug(image2)
        cam2_data = self.cameras[next_image_id]

        target = self.get_pairwise_stats(cam1_data, cam2_data)
        # target = self.get_pairwise_match_rate(cam1_data, cam2_data)
        # print(f"Pairwise match rate between {image_id} and {next_image_id}: {target}")
        # # show images side-by-side
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title(f"Image {image_id}")
        # plt.axis('off')
        # plt.imshow(image1)
        # plt.tight_layout()
        # plt.subplot(1, 2, 2)
        # plt.title(f"Image {next_image_id}")
        # plt.axis('off')
        # plt.tight_layout()
        # plt.imshow(image2)
        # plt.show()
        
        return image1_data, image2_data, target
    
    def get_pairwise_match_rate(self, cam1, cam2):
        # iterate over 3D points in cam1, find all that are also in cam2, for all pairs, compute match rate
        if not isinstance(cam1, camera) or not isinstance(cam2, camera):
            raise TypeError("Both cam1 and cam2 must be instances of the camera class.")
        else:
            total_matches = 0
            total_points = 0
            for point3D_id, point3D in cam1.points3d.items():
                if point3D_id in cam2.points3d:
                    total_matches += 1
                total_points += 1
        
            return total_matches / total_points if total_points > 0 else 0.0
    
    def get_pairwise_stats(self, cam1, cam2):
        # iterate over 3D points in cam1, find all that are also in cam2, for all pairs, compute mean angle between camera-point-camera triad
        if not isinstance(cam1, camera) or not isinstance(cam2, camera):
            raise TypeError("Both cam1 and cam2 must be instances of the camera class.")
        else:
            total_angle = 0.0
            count = 0
            for point3D_id, point3D in cam1.points3d.items():
                if point3D_id in cam2.points3d:
                    angle = angle_between_cameras(cam1, cam2, point3D["pt"].xyz)
                    total_angle += angle
                    count += 1
        
            return total_angle / count if count > 0 else 0.0

    def get_sequence(self, image_id, sequence_length):
        pass
    
    def get_label(self):
        if self.label is None:
            raise ValueError("Label not set for this scene.")
        return self.label
    
    def set_inorder(self, inorder_new):
        self.process_inorder = inorder_new

    # def reset(self):
    #     for image in self.images.values():
    #         image["used"] = False
        
    #     if self.process_inorder:
    #         self.process_order = sorted(self.images.keys())
    #     else:
    #         self.process_order = random.sample(list(self.images.keys()), len(self.images))
        
    #     self.current_index = 0

    def __len__(self):
        return len(self.images)

    def has_next(self):
        # check if all images have used set to True
        return not all(image["used"] for image in self.images.values()) and not self.current_index >= self.max_length and self.current_index < len(self)

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

        self.current_index += self.iter_rate
        self.images[next_key]["used"] = True

        return image, label, scene_id, key
    
    # def __iter__(self):
    #     self.reset()
    #     self.current_index = 0
    #     for key in self.process_order:

    #         # NOTE: add in any image processing as needed HERE
    #         image = self.img_aug(Image(self.images[key]["img"]))

    #         label = self.label

    #         yield image, label, self.scene_id, key

class FrameMatchingDataset(Dataset):
    def __init__(self, dataset_path, sample_mode="image-image", sequence_length=2, binary=True):
        self.dataset_path = Path(dataset_path)
        self.sample_mode = sample_mode
        self.sequence_length = sequence_length
        self.binary = binary

        self.scenes = []
        self.dataset_path = Path(dataset_path)
        self.mesh_json_path = Path("/home/joseph/Projects/ECS271/Project/PoolNet/redwood/meshes.json")
        self.sample_mode = sample_mode
        if self.sample_mode not in ["image-image", "sequence-image", "sequence-sequence"]:
            raise ValueError("Invalid sample_mode. Choose from 'image-image', 'sequence-image', or 'sequence-sequence'.")

        # build map of data distribution and relationships
        self.sampling_map = []
        self.lengths = []

        with open(self.mesh_json_path, 'r') as f:
            mesh_data = json.load(f)
            self.meshes = list(mesh_data)
            self.meshes = [int(id) for id in self.meshes]
        print(self.meshes)

        min_scene_size = 10
        index = 0
        for dir in self.dataset_path.iterdir():
            index += 1
            # if index < 364 and dir.is_dir() and int(dir.name) in self.meshes:
            if index < 1000 and dir.is_dir():
                print(f"Processing directory: {dir}")
                broad_file = dir / "workspace" / "datainfo.json"
                if broad_file.exists():
                    with open(broad_file, 'r') as f:
                        data_info = json.load(f)
                        primary_scene = data_info.get("best_group")

                        if primary_scene is not None:
                        
                            dir_info = dir / "workspace" / "sparse" / primary_scene / "cameras.json"
                            if dir_info.exists():
                                with open(dir_info, 'r') as f:
                                    cameras = json.load(f)
                                    if len(cameras) >= min_scene_size:
                                        # print(cameras.keys())
                                        cam_count = len(cameras)
                                        # print(f"Found {cam_count} cameras in scene {dir.name}.")
                                        scene = Scene(dir.name, dir / "workspace" / "sparse" / primary_scene)

                                        print(f"{dir.name}, {dir.name in self.meshes}")

                                        self.lengths.append(len(cameras.keys()))
                                        for image_id in cameras.keys():
                                            next_sample = {
                                                "scene" : scene,
                                                "image_id" : int(image_id),
                                            }
                                            self.sampling_map.append(next_sample)

    def get_interdataset_match(self, idx):
        im_scene = self.sampling_map[idx]["scene"]
        im_id = self.sampling_map[idx]["image_id"]
        satisfied = False
        while not satisfied:
            next_cam = im_scene.get_random_cam()
            next_image_id = next_cam.image_id
            if im_id != next_image_id:
                satisfied = True
        
        image1_data = Image.open(self.dataset_path / im_scene.cameras[im_id].fname.split("_")[0] / im_scene.cameras[im_id].fname)
        image2_data = Image.open(self.dataset_path / next_cam.fname.split("_")[0] / next_cam.fname)
        return image1_data, image2_data

    def get_intradataset_mismatch(self):
        pass

    def get_extradataset_mismatch(self, idx):
        im_scene = self.sampling_map[idx]["scene"]
        im_id = self.sampling_map[idx]["image_id"]
        satisfied = False
        while not satisfied:
            next_choice = random.choice(self.sampling_map)
            next_scene = next_choice["scene"]
            next_image_id = next_choice["image_id"]
            if im_scene.scene_id != next_scene.scene_id:
                satisfied = True
            
        image1_data = Image.open(self.dataset_path / im_scene.cameras[im_id].fname.split("_")[0] / im_scene.cameras[im_id].fname)
        image2_data = Image.open(self.dataset_path / next_scene.cameras[next_image_id].fname.split("_")[0] / next_scene.cameras[next_image_id].fname)
        return image1_data, image2_data
    
    def __len__(self):
        return len(self.sampling_map)

    def __getitem__(self, idx):
        if not self.binary:
            random_choice = random.choice([1, 2, 3])
        else:
            random_choice = random.choice([1, 3])
        if random_choice == 1:
            im1, im2 = self.get_interdataset_match(idx)
            im1 = torch.tensor(np.asarray(im1.convert("RGB")), dtype=torch.float32) / 255.0
            im2 = torch.tensor(np.asarray(im2.convert("RGB")), dtype=torch.float32) / 255.0

            # concat along new dimension
            data = torch.cat((im1.unsqueeze(0), im2.unsqueeze(0)), dim=0)  # shape (C, H, W) for both images

            return data, torch.tensor(1)
        elif random_choice == 2:
            data = self.get_intradataset_mismatch()
            return data, torch.tensor(0)
        elif random_choice == 3:
            im1, im2 = self.get_extradataset_mismatch(idx)
            im1 = torch.tensor(np.asarray(im1.convert("RGB")), dtype=torch.float32) / 255.0
            im2 = torch.tensor(np.asarray(im2.convert("RGB")), dtype=torch.float32) / 255.0

            data = torch.cat((im1.unsqueeze(0), im2.unsqueeze(0)), dim=0)  # shape (C, H, W) for both images

            return data, torch.tensor(-1)

class SequentialFrameDataset(Dataset):
    def __init__(self, dataset_path, sample_mode="image-image", sequence_length=2):
        self.scenes = []
        self.dataset_path = Path(dataset_path)
        self.mesh_json_path = Path("/home/joseph/Projects/ECS271/Project/PoolNet/redwood/meshes.json")
        self.sample_mode = sample_mode
        if self.sample_mode not in ["image-image", "sequence-image", "sequence-sequence"]:
            raise ValueError("Invalid sample_mode. Choose from 'image-image', 'sequence-image', or 'sequence-sequence'.")

        # build map of data distribution and relationships
        self.sampling_map = []
        self.lengths = []

        with open(self.mesh_json_path, 'r') as f:
            mesh_data = json.load(f)
            self.meshes = list(mesh_data)
            self.meshes = [int(id) for id in self.meshes]
        print(self.meshes)

        min_scene_size = 10
        index = 0
        for dir in self.dataset_path.iterdir():
            index += 1
            if index < 10000 and dir.is_dir() and int(dir.name) in self.meshes:
                print(f"Processing directory: {dir}")
                broad_file = dir / "workspace" / "datainfo.json"
                if broad_file.exists():
                    with open(broad_file, 'r') as f:
                        data_info = json.load(f)
                        primary_scene = data_info.get("best_group")

                        if primary_scene is not None:
                        
                            dir_info = dir / "workspace" / "sparse" / primary_scene / "cameras.json"
                            if dir_info.exists():
                                with open(dir_info, 'r') as f:
                                    cameras = json.load(f)
                                    if len(cameras) >= min_scene_size:
                                        # print(cameras.keys())
                                        cam_count = len(cameras)
                                        # print(f"Found {cam_count} cameras in scene {dir.name}.")
                                        scene = Scene(dir.name, dir / "workspace" / "sparse" / primary_scene)

                                        print(f"{dir.name}, {dir.name in self.meshes}")

                                        self.lengths.append(len(cameras.keys()))
                                        for image_id in cameras.keys():
                                            next_sample = {
                                                "scene" : scene,
                                                "image_id" : int(image_id),
                                            }
                                            self.sampling_map.append(next_sample)

        self.sequence_length = sequence_length

    def set_sequence_length(self, length):
        self.sequence_length = length

    def image_image_sample(self, scene):
        # To - within the same scene
        # Ts - feature overlap
        # Vrt - Mean angle between camera-to-feature vectors
        
        same_scene = random.choice([True, False]),
        if same_scene:
            overlap = random.choice([True, False])
            if overlap:
                # Sample two images with at least 1 shared feature
                # any scene with at least 2 images
                pass
            else:
                # any scene with < 100% image utilization
                # Sample two images with no shared features
                pass
        else:
            gt = {
                "To" : 0.0,
                "Ts" : 0.0,
                "Vrt" : 0.0,
            }

    def set_sequence_length(self, length):
        self.sequence_length = length

    def __len__(self):
        return len(self.sampling_map)
    
    def __getitem__(self, idx):
        scene = self.sampling_map[idx]["scene"]
        image_id = self.sampling_map[idx]["image_id"]

        image_1, image_2, angle = scene.get_pair(image_id)

        # print(f"image 1 shape {image_1.shape}, image 2 shape {image_2.shape}")
        # stacked_images = np.vstack((image_1, image_2))
        stacked_images = np.concatenate((image_1, image_2), axis=-1)
        stacked_images = torch.tensor(stacked_images, dtype=torch.float32)
        stacked_images = stacked_images.permute(2, 0, 1)  # Change to (C, H, W) format
        # print("angle", angle)
        # print("stacked_images shape", stacked_images.shape)

        # print(f"pixel range min: {stacked_images.min()}, max: {stacked_images.max()}")

        stacked_images = stacked_images / 255.0  # Normalize to [0, 1] range
        return (stacked_images, torch.tensor(angle))

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
    dataset = SequentialFrameDataset(data_dir)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # get sample scene from dataloader
    for i, scene in enumerate(dataloader):
        for sample in scene:
            image, label, scene_id, key = sample
            print(f"Scene ID: {scene_id}, Key: {key}, Image: {image}, Label: {label}")