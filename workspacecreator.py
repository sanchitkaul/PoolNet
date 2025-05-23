import os
from pathlib import Path

# MODIFY THESE PATHS BASED ON YOUR STRUCTURE
SOURCE_ROOT = os.path.expanduser('/content/workspace')
  # folder containing folders like 2c5537eddf
WORKSPACE_ROOT = 'workspace'  # where COLMAP workspaces will be created

os.makedirs(WORKSPACE_ROOT, exist_ok=True)

video_folders = [f for f in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, f))]

for video_id in video_folders:
    source_video_path = os.path.join(SOURCE_ROOT, video_id)
    target_workspace_path = os.path.join(WORKSPACE_ROOT, video_id)
    images_path = os.path.join(target_workspace_path, 'images')

    os.makedirs(images_path, exist_ok=True)

    print(f"Setting up workspace for {video_id}")

    for file in os.listdir(source_video_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.abspath(os.path.join(source_video_path, file))
            dst = os.path.join(images_path, file)

            # Create symlink to save disk space
            if not os.path.exists(dst):
                os.symlink(src, dst)

print("✅ All COLMAP workspaces set up under:", WORKSPACE_ROOT)