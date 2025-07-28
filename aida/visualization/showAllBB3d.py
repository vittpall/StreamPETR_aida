import torch 
import json 
import numpy as np 
from PIL import Image 
from mmdet3d.core.bbox import LiDARInstance3DBoxes 
from scipy.spatial.transform import Rotation as R 
from visualize_3d_carlo import show_bboxes_3d 

import pickle 
import matplotlib.pyplot as plt 
import open3d as o3d 
import numpy as np #Wayland is not support, use X11 xorg 
import os 
os.environ["XDG_SESSION_TYPE"] = "x11" 
os.environ["GDK_BACKEND"] = "x11" 
# === getting info from the .pkl  === 

with open('../submission/pts_bbox/val_fine_tuning_streampetr.pkl', 'rb') as f: data = pickle.load(f)
#with open('../submission/streamPETR/results.pkl', 'rb') as f:
#    data = pickle.load(f)


# === Pick the sample by token ===
for i in range(40, 41): #Modify depending on the number of frames
    #pcd_path = f'../lidar/2025-03-13_11-54_{i:06d}.pcd'

    #Working pcd
    pcd_path = f'../lidar/2025-03-20_12-25_000000.pcd'
    # === Pick the sample by token ===
    #sample_token = f"2025-03-13_11-54_{i:06d}"
    sample_token = f"2025-03-27_12-04_normal_000365"
    #sample_token = f"2025-03-27_12-04_normal_0000332"
    #info = next(i for i in data['infos'] if i['token'] == sample_token)
    info = next(info for info in data['infos'] if info['token'] == sample_token)
    cam_info = info['cams']['CAM_FRONT']  # choose camera

    # === Get intrinsics (P) ===
    cam_intrinsic = np.array(cam_info['cam_intrinsic'])  # 3x3
    P = np.eye(3, 4)
    P[:3, :3] = cam_intrinsic
    P = torch.tensor(P, dtype=torch.float32)

    # === Get extrinsics: LiDAR → Camera ===
    # The stored values are camera → LiDAR, so invert them

    R_cam2lidar = np.array(cam_info['sensor2lidar_rotation'])  # 3x3
    t_cam2lidar = np.array(cam_info['sensor2lidar_translation'])  # 3x1
    
    print("calibration", R_cam2lidar)
    print("calibration2:", t_cam2lidar)
    # Invert the transform to get LiDAR → Camera
    R_lidar2cam = R_cam2lidar.T
    t_lidar2cam = -R_lidar2cam @ t_cam2lidar

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :3] = R_lidar2cam
    Tr_velo_to_cam[:3, 3] = t_lidar2cam
    Tr_velo_to_cam = torch.tensor(Tr_velo_to_cam, dtype=torch.float32)

    # === Rectification matrix (identity) ===
    R_rect = torch.eye(4)

    # === MMDet bboxes ===

    with open("../submission/streamPETR/results_FineTuning3ep.pkl", "rb") as f:
        results = pickle.load(f)

    # Read the .pcd (ASCII or binary) into an Open3D PointCloud
    pcd_o3d = o3d.io.read_point_cloud(pcd_path)

    # Extract Nx3 pts, then append a 1 for homogeneous coords ⇒ Nx4
    pts = np.asarray(pcd_o3d.points)                   # shape (N,3)
    scan = np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1).astype(np.float32)  # shape (N,4)


    results_filtered = []

    filter_bboxes = results[i]['scores_3d'] > 0.3
    #print(filter_bboxes.shape)
    #bboxes_3d_manual = results[0]['boxes_3d'][filter_bboxes]
    # Slice and re-wrap the bounding boxes correctly
    boxes_tensor = results[i]['boxes_3d'][filter_bboxes].tensor[:, :7]
    bboxes_3d_manual = LiDARInstance3DBoxes(boxes_tensor)

    print(bboxes_3d_manual)

    labels_3d_manual = results[i]['labels_3d'][filter_bboxes].numpy()

    color_dict = {
        0: (255, 255, 0),   # Yellow
        1: (0, 255, 0),     # Green
        2: (255, 0, 0),     # Red
        6: (0, 255, 255),   # Cyan
        7: (255, 0, 255),   # Magenta
        8: (255, 165, 0),   # Orange
        9: (128, 0, 255)    # Violet/Purple
    }

    color_dict = {
    0: (255/255.0, 255/255.0, 0/255.0),   # Yellow
    1: (0/255.0, 255/255.0, 0/255.0),     # Green
    2: (255/255.0, 0/255.0, 0/255.0),     # Red
    3: (0/255.0, 0/255.0, 255/255.0),     # Blue (Added: Example color for label 3)
    4: (255/255.0, 128/255.0, 0/255.0),   # Orange-ish (Added: Example color for label 4)
    5: (128/255.0, 0/255.0, 128/255.0),   # Purple (Added: Example color for label 5)
    6: (0/255.0, 255/255.0, 255/255.0),   # Cyan
    7: (255/255.0, 0/255.0, 255/255.0),   # Magenta
    8: (255/255.0, 165/255.0, 0/255.0),   # Orange
    9: (128/255.0, 0/255.0, 255/255.0)    # Violet/Purple
    # Add any other missing label keys from your 'labels_3d' tensor here
    }

    calibration = {
        'Tr_velo_to_cam': Tr_velo_to_cam.numpy()
    }

    # === Visualize ===
    output_image = show_bboxes_3d(
        point_cloud=scan, calibration_data=calibration,
        bboxes_3d=bboxes_3d_manual,
        labels=labels_3d_manual,
        colors_dict=color_dict,
        save_capture=True,
        adjust_position=True,
        show_window=False
    )

    # === Save the output image ===
    # Convert numpy array to PIL Image and save
    if output_image is not None:
        pil_image = Image.fromarray(output_image)
        pil_image.save(f"./boxes/output_{i:06d}.png")

    else:
        print("No image was captured - output_image is None")


 
