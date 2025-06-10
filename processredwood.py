from pathlib import Path
import json
import pandas as pd
import numpy as np
import subprocess
import ffmpeg
import time
from colmaprun import compute_metrics, parse_images_txt, convert_model_to_text, compute_metrics_updated
import os
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def process_colmap(workspace_dir, output_dir):
    start_time = time.time()
    subprocess.run([
        'colmap', 'automatic_reconstructor',
        '--workspace_path', workspace_dir,
        '--image_path', output_dir,
        '--use_gpu', '1'
    ], check=True)
    end_time = time.time()

    print(f"COLMAP reconstruction completed in {end_time - start_time:.2f} seconds.")

    return end_time - start_time

def process_glomap(workspace_dir, output_dir):
    start_time = time.time()
    subprocess.run([
        'colmap', 'feature_extractor',
        '--image_path', output_dir,
        '--database_path', workspace_dir / 'database.db',
        '--SiftExtraction.use_gpu', '1',
    ])
    feature_extractor_time = time.time() - start_time

    subprocess.run([
        'colmap', 'exhaustive_matcher',
        '--database_path', workspace_dir / 'database.db',
    ], check=True)
    matcher_time = time.time() - feature_extractor_time

    subprocess.run([
        'glomap', 'mapper',
        '--database_path', workspace_dir / 'database.db',
        '--image_path', output_dir,
        '--output_path', workspace_dir / 'sparse',
    ], check=True)
    end_time = time.time()

    print(f"GLOMAP reconstruction completed in {end_time - start_time:.2f} seconds.")
    print(f"Feature extraction time: {feature_extractor_time - start_time:.2f} seconds.")
    print(f"Matcher time: {matcher_time - start_time:.2f} seconds.")
    

def main():
    redwood_path = Path("/media/SharedStorage/redwood/mp4")
    output_path = Path("/media/SharedStorage/redwood/output")
    category_json_path = Path("./redwood/categories.json")

    if not redwood_path.exists():
        print(f"Redwood path {redwood_path} does not exist. Please check the path.")
        return
    else:
        # setup data loading
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"output path: {output_path} did not exist, but has been created.")
        
        # iterate over all data
        j = 0
        # for video_file in sorted(redwood_path.glob("*.mp4"), key=lambda f: f.stat().st_size):
        for video_file in redwood_path.glob("*.mp4"):
            # if j == 0:
            #     j+= 1
            #     continue
            print(f"Processing file: {video_file.name}")
            dir_name = video_file.name.split(".")[0]
            output_dir = output_path / dir_name
            workspace_dir = output_dir / "workspace"

            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"output directory does not exist, created output directory: {output_dir}")
            
                if not workspace_dir.exists():
                    workspace_dir.mkdir(parents=True, exist_ok=True)
                    print(f"workspace directory does not exist, created workspace directory: {workspace_dir}")

                # run ffmpeg to extract individual frames to run on 
                output_pattern = str(output_dir / f"{dir_name}_%04d.png")
                frame_rate = 5

                ffmpeg.input(str(video_file)).output(
                    output_pattern,
                    vf=f"fps={frame_rate}",
                    start_number=0,
                ).run()

                # run SFM software on the extracted frames
                # process_glomap(workspace_dir, output_dir)
                try:
                    process_colmap(workspace_dir, output_dir)
                except Exception as e:
                    print(f"Error processing {video_file.name}: {e}")
                    continue
            else:
                print(f"Output directory {output_dir} already exists, skipping processing for {video_file.name}.")

                # temporarily run only the first instance to test. Remove return statement to process all videos

class camera:
    def __init__(self, image_id, qvec, tvec, cam_id, fname):
        
        self.image_id = image_id
        self.qvec = qvec
        self.tvec = tvec
        self.cam_id = cam_id
        self.fname = fname
        self.points = []
        self.points3d = {}

    def add_point(self, point_obj):
        self.points.append(point_obj)
    
    def add_point3D(self, point3D_id, point, point2d_idx):
        if point3D_id not in self.points3d.keys():
            # print(f"Adding Point3D ID {point3D_id} to camera {self.image_id} at point2d index {point2d_idx}.")
            self.points3d[point3D_id] = {'pt' : point, 'point2d_idx' : point2d_idx}
        else:
            pass
            # print(f"Warning: Point3D ID {point3D_id} already exists in camera {self.image_id} duplicates at {point2d_idx} and {self.points3d[point3D_id]}. Skipping addition.")
            # raise ValueError(f"Point3D ID {point3D_id} already exists in camera {self.image_id}. Cannot add duplicate.")
        
    def to_dict(self):
        return cameraEncoder().default(self)
        
class cameraEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, camera):
            return {
                'image_id': obj.image_id,
                'qvec': obj.qvec,
                'tvec': obj.tvec,
                'cam_id': obj.cam_id,
                'fname': obj.fname,
                'points': [point.to_dict() for point in obj.points],
                'points3d': {point3D_id: {'pt': point['pt'].to_dict(), 'point2d_idx': point['point2d_idx']} for point3D_id, point in obj.points3d.items()}
            }
        return super().default(obj)

class point2D:
    def __init__(self, point_idx, x, y):
        self.point_idx = point_idx
        self.x = x
        self.y = y

    def to_dict(self):
        return point2DEncoder().default(self)
        
class point2DEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, point2D):
            return {
                'point_idx': obj.point_idx,
                'x': obj.x,
                'y': obj.y
            }
        return super().default(obj)

class point3D:
    def __init__(self, point3D_id, xyz, rgb, error):
        self.point3D_id = point3D_id
        self.xyz = xyz  # [x, y, z]
        self.rgb = rgb  # [r, g, b]
        self.error = error
        self.image_pointidx = []

    def add_image_point(self, image_id, point_idx):
        self.image_pointidx.append((image_id, point_idx))

    def to_dict(self):
        return point3DEncoder().default(self)

class point3DEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, point3D):
            return {
                'point3D_id': obj.point3D_id,
                'xyz': obj.xyz,
                'rgb': obj.rgb,
                'error': obj.error,
                'image_pointidx': obj.image_pointidx
            }
        return super().default(obj)

def cam_center_from_qvec(quat, tvec):
    return (R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix().T @ -np.array(tvec).reshape(3, 1)).flatten()

def angle_between_cameras(cam1, cam2, p):
    vec1 = cam_center_from_qvec(cam1.qvec, cam1.tvec) - p
    vec2 = cam_center_from_qvec(cam2.qvec, cam2.tvec) - p

    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    return np.degrees(np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0)))

def get_camera_data():
    pass
        
def get_cam_points(sparse_path):
    cameras = {}
    points3D = {}

    images_txt = sparse_path / "images.txt"
    points3D_txt = sparse_path / "points3D.txt"

    print(f"Processing cameras from {images_txt} and points3D from {points3D_txt}")

    if images_txt.exists():
        with open(images_txt, 'r') as f:
            # skip first 3 lines
            for _ in range(3):
                next(f)
            intro = next(f)

            cam_line = True
            cur_image_id = None
            for line in f:
                if cam_line:
                    image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, fname = line.strip().split(" ")
                    image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, fname = int(image_id), float(qw), float(qx), float(qy), float(qz), float(tx), float(ty), float(tz), int(cam_id), fname
                    cur_image_id = image_id
                    cameras[image_id] = camera(image_id, [qw, qx, qy, qz], [tx, ty, tz], cam_id, fname)
                    cam_line = False
                else:
                    point2D_raw = line.strip().split(" ")
                    if len(point2D_raw) % 3 != 0:
                        print(f"Skipping malformed line: {line.strip()}")
                    else:
                        for i in range(0, len(point2D_raw), 3):
                            x = float(point2D_raw[i])
                            y = float(point2D_raw[i + 1])
                            cameras[cur_image_id].add_point(point2D(i // 3, x, y))
                    cam_line = True

    if points3D_txt.exists():
        with open(points3D_txt, 'r') as f:
            # skip first 2 lines
            for _ in range(2):
                next(f)

            intro = next(f)

            for line in f:
                point3D_raw = line.strip().split(" ")
                point3D_id, x, y, z, r, g, b, error = point3D_raw[:8]
                point3D_id, x, y, z, r, g, b, error = int(point3D_id), float(x), float(y), float(z), int(r), int(g), int(b), float(error)
                point3D_obj = point3D(point3D_id, [x, y, z], [r, g, b], error)
                point2D_raw = point3D_raw[8:]
                if len(point2D_raw) % 2 != 0:
                    print(f"Skipping malformed line: {line.strip()}")
                else:
                    for i in range(0, len(point2D_raw), 2):
                        image_id = int(point2D_raw[i])
                        point_idx = int(point2D_raw[i + 1])
                        cameras[image_id].add_point3D(point3D_id, point3D_obj, point_idx)
    
    # save camera dict
    with open(sparse_path / "cameras.json", 'w') as f:
        dump_obj = {id: cam.to_dict() for id, cam in cameras.items()}
        json.dump(dump_obj, f, indent=4)
    
    return cameras, points3D

def generate_targets():
    data_path = Path("/media/SharedStorage/redwood/output")

    if not data_path.exists():
        print(f"Data path {data_path} does not exist. Please check the path.")
        return
    else:
        data_distribution = {
            "ts0" : 0,
            "ts1" : 0,
        }
        sizes = {
            "high" : 0,
            "low" : 0,
            "zero" : 0,
        }
        for database in data_path.glob("*"):
            if database.is_dir():
                # print(f"Processing database: {database.name}")

                workspace = database / "workspace"
                sparse_path = workspace / "sparse"
                dense_path = workspace / "dense"
                db_path = workspace / "database.db"
                excepted = False

                output_label = {
                    'video_id': str(database.name),
                    'Ts': 0,
                    'To': 0,
                    'Vr': 0,
                    'Vt': 0,
                }
                total_cameras = len([f for f in os.listdir(database) if f.endswith(('.jpg', '.png', '.jpeg'))])
                if sparse_path.exists():
                    sparse_subdirs = [d for d in sparse_path.iterdir() if d.is_dir()]
                    # print(f"Found subdirs for sparse_path with {total_cameras} and {len(sparse_subdirs)} subdirs in {sparse_path}")
                    max_cam_count = -1
                    max_cam_dir = None
                    subdir_size_save = 0
                    cam_count_save = 0
                    for subdir in sparse_subdirs:
                        subdir_size = sum(f.stat().st_size for f in subdir.glob('*') if f.is_file())
                        with open(subdir / "cameras.txt", 'r') as f:
                            cam_count = int([str(next(f)) for _ in range(3)][-1].split(" ")[-1])
                        if cam_count > max_cam_count:
                            max_cam_count = cam_count
                            max_cam_dir = subdir
                            subdir_size_save = subdir_size
                            cam_count_save = cam_count
                        
                        # print(f"\t\tSubdir: {subdir.name}, size: {subdir_size / (1024 * 1024):.2f} MB, cameras: {cam_count}")
                        txt_path = subdir / "images.txt"
                        if txt_path.exists():
                            # already converted
                            pass
                        else:
                            try:
                                convert_model_to_text(subdir)
                                print(f"Converted {subdir} to text format.")
                            except Exception as e:
                                print(f"Error converting {subdir} to text format: {e}")
                                excepted = True
                    
                    # print(f"\t\tSubdir: {max_cam_dir}, size: {subdir_size_save / (1024 * 1024):.2f} MB, cameras: {cam_count_save}, match rate {cam_count_save / total_cameras:.2f}")
                    if max_cam_dir is not None:
                        get_cam_points(max_cam_dir)
                        dataset_dict = {
                            "matched" : True,
                            "best_group" : max_cam_dir.name,
                            "num_cams" : max_cam_count,
                            "total_cams" : total_cameras,
                        }
                        save_file = workspace / "datainfo.json"
                        with open(save_file, 'w') as f:
                            json.dump(dataset_dict, f, indent=4)
                    else:
                        print(f"No valid subdirectory found in {sparse_path} for {database.name}.")
                        sizes["zero"] += 1
                        excepted = True
                        dataset_dict = {
                            "matched" : False,
                            "best_group" : None,
                            "num_cams" : 0,
                            "total_cams" : total_cameras,
                        }
                        save_file = workspace / "datainfo.json"
                        with open(save_file, 'w') as f:
                            json.dump(dataset_dict, f, indent=4)
                    
                    if cam_count_save / total_cameras > 0.40:
                        sizes["high"] += 1
                        ts = 1
                        to = cam_count_save / total_cameras
                    else:
                        ts = 0
                        to = cam_count_save / total_cameras
                        sizes["low"] += 1
                else:
                    # print(f"sparse_path {sparse_path} does not exist, skipping {database.name}")
                    sizes["zero"] += 1
                    pass

                # if workspace.exists() and sparse_path.exists() and dense_path.exists() and db_path.exists():
                #     try:
                #         # check if 0, 1, or 2 exists in sparse_path
                #         max_sparse = None
                #         for i in range(2, -1, -1):
                #             if (sparse_path / str(i)).exists():
                #                 max_sparse = i
                #                 break
                        
                #         if max_sparse is not None:
                #             sparse_model_path = os.path.join(sparse_path, str(max_sparse))
                #             convert_model_to_text(sparse_model_path)
                #             images_txt = os.path.join(sparse_model_path, "images.txt")
                #             # print(images_txt)
                #             poses = parse_images_txt(images_txt)

                #             total_frames = len([f for f in os.listdir(database) if f.endswith(('.jpg', '.png', '.jpeg', '.png'))])

                #             # ts, to, vr, vt_raw = compute_metrics(poses, total_frames)
                #             ts, to, vr, vt_raw = compute_metrics_updated(poses, total_frames)
                #             vt = round(np.log1p(vt_raw), 3)
                #             # print(f"✓ {database.name}: Ts={ts}, To={to}, Vr={vr}, Vt={vt}")
                #             if ts == 1:
                #                 data_distribution["ts1"] += 1
                #                 output_label = {
                #                     'video_id': str(database.name),
                #                     'Ts': ts,
                #                     'To': to,
                #                     'Vr': vr,
                #                     'Vt': vt,
                #                 }
                #             elif ts == 0:
                #                 data_distribution["ts0"] += 1
                #     except Exception as e:
                #         print(f"Error processing {database.name}: {e}")
                #         excepted = True
                # else:
                #     data_distribution["ts0"] += 1
                
                # save
                # json_path = database / "label.json"
                # with open(json_path, 'w') as f:
                #     json.dump(output_label, f, indent=4)
                    # print(f"Saved label to {json_path}")
        print(f"Data distribution: {data_distribution}")
        print(f"Size distribution: {sizes}")

def extract_frames_only(video_dir, output_dir, processed_dir, frame_rate=5, json_data_desc="./meshes.json"):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    json_data_desc = Path(json_data_desc)
    processed_dir = Path(processed_dir)

    
    pre_processed = {} # dictionary is faster access time than list
    for i, scene_dir in enumerate(os.listdir(processed_dir)):
        pre_processed[int(scene_dir)] = True

    meshable = json_data_desc
    with open(meshable, 'r') as f:
        meshable_data = json.load(f)
        meshable_data = list(meshable_data)
        meshable_data = [int(x) for x in meshable_data]

    for i, video_file in tqdm(enumerate(os.listdir(video_dir))):
        video_file_int = int(video_file.split(".")[0])
        if video_file_int not in pre_processed and video_file_int in meshable_data:
            if not os.path.exists(output_dir / video_file.split(".")[0]):
                os.makedirs(output_dir / video_file.split(".")[0], exist_ok=True)
                output_pattern = str(output_dir / f"{video_file.split('.')[0]}" / f"{video_file}_%04d.png")
                try:
                    ffmpeg.input(str(video_dir / f"{video_file.split('.')[0]}.mp4")).output(
                        output_pattern,
                        vf=f"fps={frame_rate}",
                        start_number=0,
                    ).global_args('-loglevel', 'quiet').run()
                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
            else:
                print(f"Output directory {output_dir / video_file.split('.')[0]} already exists, skipping processing for {video_file}.")

if __name__ == "__main__":
    # main()
    # generate_targets()
    extract_frames_only(
        video_dir="/media/SharedStorage/redwood/mp4", 
        output_dir="/media/SharedStorage/redwood/FramesOnly", 
        processed_dir="/media/SharedStorage/redwood/output",
        frame_rate=5, 
        json_data_desc="/home/joseph/Projects/ECS271/Project/redwood-3dscan/meshes.json")