# test_romp_inference.py

import os
import shutil
import cv2
import argparse
import torch
import numpy as np
import romp
from romp.main import default_settings
from tqdm import tqdm
import trimesh
import pyrender

def visualize_mesh(vertices, faces):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    scene = pyrender.Scene()
    mesh_obj = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_obj)
    pyrender.Viewer(scene, use_raymond_lighting=True)

def main(video_path, num_frames, frames_dir='frames'):
    # 1) Extract frames (always clear out any old ones first)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    fps = num_frames / 5.0  # assuming a 5 s video
    os.system(
        f'ffmpeg -i "{video_path}" -vf fps={fps} '
        f'{frames_dir}/frame_%03d.jpg'
    )

    # 2) Load & prepare SMPL face indices once
    smpl_pth = default_settings.smpl_path  # e.g. ~/.romp/SMPL_NEUTRAL.pth
    smpl_data = torch.load(smpl_pth, map_location='cpu')
    faces = smpl_data['f']
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    faces = faces.reshape(-1, 3).astype(np.int64)

    # 3) Set up ROMP for inference
    settings = default_settings
    settings.mode = 'video'
    settings.calc_smpl = True
    settings.render_mesh = False
    settings.show = False
    model = romp.ROMP(settings)

    # 4) Loop over sampled frames
    img_files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))[:num_frames]
    for i, img_name in enumerate(tqdm(img_files, desc='Running ROMP')):
        img = cv2.imread(os.path.join(frames_dir, img_name))  # BGR
        out = model(img)

        verts = out['verts']
        if isinstance(verts, torch.Tensor):
            verts = verts.cpu().numpy()

        # ROMP returns a batch of size 1: squeeze it away
        if verts.ndim == 3 and verts.shape[0] == 1:
            verts = verts[0]

        # Sanity‐check on the first frame
        if i == 0:
            print(f"[INFO] verts shape: {verts.shape}, faces shape: {faces.shape}")
            # you should now see verts shape: (6890, 3), faces shape: (13776, 3)

        visualize_mesh(verts, faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sample frames & run ROMP inference"
    )
    parser.add_argument('-v', '--video', required=True,
                        help="Path to your T‑pose MP4")
    parser.add_argument('-n', '--num_frames', type=int, default=10,
                        help="Number of frames to sample")
    parser.add_argument('-f', '--frames_dir', default='frames',
                        help="Directory for extracted frames")
    args = parser.parse_args()
    main(args.video, args.num_frames, args.frames_dir)
