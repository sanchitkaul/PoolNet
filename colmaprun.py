import os
import subprocess
import numpy as np
import pandas as pd

WORKSPACE_ROOT = os.path.expanduser('/content/workspace')  # modify if needed
RESULTS_CSV = 'sfm_results.csv'
min_success_views = 3  # Ts = 1 if at least 3 frames reconstructed

def convert_model_to_text(sparse_model_path):
    subprocess.run([
        'colmap', 'model_converter',
        '--input_path', sparse_model_path,
        '--output_path', sparse_model_path,
        '--output_type', 'TXT'
    ], check=True)

def parse_images_txt(images_txt_path):
    poses = []
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) >= 10:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                poses.append((qw, qx, qy, qz, tx, ty, tz))
    return poses

def compute_metrics(poses, total_frames):
    used = len(poses)
    ts = 1 if used >= min_success_views else 0
    to = used / total_frames if total_frames else 0

    translations = np.array([p[4:] for p in poses])
    vt = np.linalg.norm(translations.max(axis=0) - translations.min(axis=0)) if len(translations) >= 2 else 0

    rotations = np.array([p[:4] for p in poses])
    angles = []
    for i in range(1, len(rotations)):
        dot = np.abs(np.dot(rotations[0], rotations[i]))
        angle = 2 * np.arccos(np.clip(dot, -1, 1))
        angles.append(np.degrees(angle))
    vr = np.mean(angles) / 180 if angles else 0

    return ts, round(to, 3), round(vr, 3), round(vt, 3)
if __name__ == "__main__":
    results = []

    video_folders = [f for f in os.listdir(WORKSPACE_ROOT) if os.path.isdir(os.path.join(WORKSPACE_ROOT, f))]

    for vid in video_folders:
        print(f"\n▶ Processing: {vid}")
        folder = os.path.join(WORKSPACE_ROOT, vid)
        images_path = os.path.join(folder, 'images')
        sparse_path = os.path.join(folder, 'sparse')
        total_frames = len([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

        try:
            # COLMAP one-liner
            subprocess.run([
                'colmap', 'automatic_reconstructor',
                '--workspace_path', folder,
                '--image_path', images_path,
                '--use_gpu', '1'
            ], check=True)


            # Convert binary to TXT
            sparse_model_path = os.path.join(sparse_path, '0')
            convert_model_to_text(sparse_model_path)

            # Parse images.txt
            images_txt = os.path.join(sparse_model_path, 'images.txt')
            poses = parse_images_txt(images_txt)

            # Compute metrics
            ts, to, vr, vt = compute_metrics(poses, total_frames)
            print(f"✓ {vid}: Ts={ts}, To={to}, Vr={vr}, Vt={vt}")

            results.append({
                'video_id': vid,
                'Ts': ts,
                'To': to,
                'Vr': vr,
                'Vt': vt
            })

        except Exception as e:
            print(f"❌ Failed on {vid}: {e}")
            results.append({
                'video_id': vid,
                'Ts': -1,
                'To': -1,
                'Vr': -1,
                'Vt': -1
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ Results saved to {RESULTS_CSV}")
