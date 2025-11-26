# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
import glob
import pickle
import cv2
import numpy as np
from torch.utils.data import Dataset
from downstreams.utils.geometry import closed_form_inverse_se3
from downstreams.datasets.dataset_util import *
from downstreams.datasets.base_dataset import BaseDataset
import torch

class Co3dDataset(BaseDataset):
    def __init__(
        self,
        len_train: int = 50000,
        len_test: int = 5000,
        split: str = "train",
        expand_ratio = 4,
        Co3d_DIR: str = "",
    ):
        """
        Initialize the ARDataset.

        Args:
            split (str): Dataset split, either 'train' or 'test'.
            AR_DIR (str): Directory path to AR data.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super(Co3dDataset, self).__init__(len_train, len_test, split, expand_ratio)

        self.Co3d_DIR = Co3d_DIR
        logging.info(f"Co3d_DIR is {self.Co3d_DIR}")

        with open(osp.join(self.Co3d_DIR, 'co3d_valid_seqs.txt'), "r") as f:
            seq_list = [line.strip() for line in f.readlines()]

        self.sequence_list = seq_list
        self.sequence_list_len = len(self.sequence_list)

        logging.info(f"{self.status}: Co3D Real Data size: {self.sequence_list_len}")
        logging.info(f"{self.status}: Co3D Data dataset length: {len(self)}")

    def _get_impath(self, seq_name, view_idx):
        return osp.join(self.Co3d_DIR, seq_name, 'images', f'frame{view_idx:06n}.jpg')

    def _get_depthpath(self, seq_name, view_idx):
        return osp.join(self.Co3d_DIR, seq_name, 'depths', f'frame{view_idx:06n}.jpg.geometric.png')

    def _get_maskpath(self, seq_name, view_idx):
        return osp.join(self.Co3d_DIR, seq_name, 'masks', f'frame{view_idx:06n}.png')

    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])
        return depthmap

    def _get_metadatapath(self, seq_name, view_idx):
        return osp.join(self.Co3d_DIR, seq_name, 'images', f'frame{view_idx:06n}.npz')

    def __getitem__(
        self,
        seq_index: int = None,
        img_per_seq: int = 8,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        seq_index = seq_index % self.sequence_list_len
        seq_name = self.sequence_list[seq_index % self.sequence_list_len]
        img_dir = os.path.join(self.Co3d_DIR, seq_name, 'images')
        available_images = sorted(glob.glob(osp.join(img_dir, "*.jpg")))
        num_images = len(available_images)
        all_frame_nums = [int(osp.splitext(osp.basename(p))[0].replace('frame', '')) for p in available_images]

        ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)
        # if self.get_nearby:
        #     ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)


        target_image_shape = np.array([224, 224])
        images = []
        images_paths = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for idx in ids:
            view_index = all_frame_nums[idx]
            
            impath = self._get_impath(seq_name, view_index)
            depthpath = self._get_depthpath(seq_name, view_index)
            metadata_path = self._get_metadatapath(seq_name, view_index)
            basename = osp.basename(impath)

            # image
            image = imread_cv2(impath)

            # intrinscis and extrinsics (opencv format)
            input_metadata = np.load(metadata_path)
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            extri_opencv = closed_form_inverse_se3(camera_pose[None])[0][:3, :] # w2c
            intri_opencv = input_metadata['camera_intrinsics'].astype(np.float32)
            
            # depth map
            depth_map = self._read_depthmap(depthpath, input_metadata)

            # load object mask
            maskpath = self._get_maskpath(seq_name, view_index)
            maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            maskmap = (maskmap / 255.0) > 0.1
            depth_map *= maskmap

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=impath,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            images_paths.append(impath)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "Co3D"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "images_paths": images_paths,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        sample = self.post_processing(batch)

        return sample

from downstreams.utils.train_utils import normalize_camera_extrinsics_and_points_batch
def process_batch(batch):      
    normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths, local_points = \
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
        )
    # Replace the original values in the batch with the normalized ones.
    batch["extrinsics"] = normalized_extrinsics
    batch["cam_points"] = normalized_cam_points
    batch["world_points"] = normalized_world_points
    batch["depths"] = normalized_depths
    batch["local_points"] = local_points
    return batch

def save_ply(points, colors, filename):
    import open3d as o3d                
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

from torch.utils.data import default_collate 
if __name__ == "__main__":
    dataste = Co3dDataset(Co3d_DIR='/data8T/co3d_processed/')
    sample = dataste[500]
    batch = default_collate([sample]) 
    processed_batch = process_batch(batch)
    save_ply(
        processed_batch["world_points"][0].reshape(-1, 3), 
        processed_batch["images"][0].permute(0, 2, 3, 1).reshape(-1, 3), 
        "debug.ply"
    )
    print(1)