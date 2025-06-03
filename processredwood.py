from pathlib import Path
import json
import pandas as pd
import numpy as np
import subprocess
import ffmpeg
import time
from colmaprun import compute_metrics, parse_images_txt, convert_model_to_text
import os

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
        for database in data_path.glob("*"):
            if database.is_dir():
                print(f"Processing database: {database.name}")

                workspace = database / "workspace"
                sparse_path = workspace / "sparse"
                dense_path = workspace / "dense"
                db_path = workspace / "database.db"
                excepted = False

                output_label = {
                    'video_id': str(database.name),
                    'Ts': -1,
                    'To': -1,
                    'Vr': -1,
                    'Vt': -1,
                }

                if workspace.exists() and sparse_path.exists() and dense_path.exists() and db_path.exists():
                    try:
                        # check if 0, 1, or 2 exists in sparse_path
                        max_sparse = None
                        for i in range(2, -1, -1):
                            if (sparse_path / str(i)).exists():
                                max_sparse = i
                                break
                        
                        if max_sparse is not None:
                            sparse_model_path = os.path.join(sparse_path, str(max_sparse))
                            convert_model_to_text(sparse_model_path)
                            images_txt = os.path.join(sparse_model_path, "images.txt")
                            poses = parse_images_txt(images_txt)
                            print(poses)

                            total_frames = len([f for f in os.listdir(database) if f.endswith(('.jpg', '.png', '.jpeg', '.png'))])

                            ts, to, vr, vt = compute_metrics(poses, total_frames)
                            print(f"✓ {database.name}: Ts={ts}, To={to}, Vr={vr}, Vt={vt}")
                            if ts == 1:
                                data_distribution["ts1"] += 1
                                output_label = {
                                    'video_id': str(database.name),
                                    'Ts': ts,
                                    'To': to,
                                    'Vr': vr,
                                    'Vt': vt,
                                }
                            elif ts == 0:
                                data_distribution["ts0"] += 1
                    except Exception as e:
                        print(f"Error processing {database.name}: {e}")
                        excepted = True
                else:
                    data_distribution["ts0"] += 1
                
                # save
                json_path = database / "label.json"
                with open(json_path, 'w') as f:
                    json.dump(output_label, f, indent=4)
                    print(f"Saved label to {json_path}")
        print(f"Data distribution: {data_distribution}")


if __name__ == "__main__":
    # main()
    generate_targets()