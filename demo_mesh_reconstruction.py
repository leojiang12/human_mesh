#!/usr/bin/env python
import os, sys, cv2
import torch
import numpy as np
import trimesh, pyrender

# ROMP imports
from romp.runner import Runner

def sample_frames(video_path, out_dir, num_frames=8):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total-1, num_frames, dtype=int)
    saved = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret: continue
        path = os.path.join(out_dir, f"frame_{i:04d}.png")
        cv2.imwrite(path, frame)
        saved.append(path)
    cap.release()
    return saved

def reconstruct_mesh(frame_paths, device='cuda'):
    runner = Runner(checkpoint_path="checkpoints/romp_pretrained.pth",
                    device=device)
    meshes = []
    for img_path in frame_paths:
        mesh = runner.run_on_image(img_path)  # returns trimesh.Trimesh
        meshes.append(mesh)
    return meshes

def visualize_mesh(mesh):
    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_node)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reconstruct_and_viz.py input.mp4")
        sys.exit(1)

    video = sys.argv[1]
    frame_dir = "sampled_frames"
    print("▶ Sampling frames…")
    frames = sample_frames(video, frame_dir, num_frames=8)

    print("▶ Reconstructing…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meshes = reconstruct_mesh(frames, device=device)

    # Save the first mesh as OBJ
    out_obj = "recon0.obj"
    meshes[0].export(out_obj)
    print(f"✔ Mesh saved to {out_obj}")

    print("▶ Visualizing first mesh…")
    visualize_mesh(meshes[0])
