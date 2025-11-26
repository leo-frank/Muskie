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
from natsort import natsorted
import torch
import cv2
import numpy as np
from downstreams.utils.geometry import closed_form_inverse_se3
from torch.utils.data import Dataset
from downstreams.datasets.dataset_util import *
from downstreams.datasets.base_dataset import BaseDataset



seq_list = [
    "breakfast_room", "complete_kitchen", "green_room", "grey_white_room", "kitchen", "morning_apartment", "staircase", "thin_geometry", "whiteroom"
]

class NRGBDMultiviewDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        NRGBDMultiview_DIR: str = "",
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the NRGBDMultiviewDataset.
        """
        super(NRGBDMultiviewDataset, self).__init__(len_train, len_test, split, expand_ratio)

        self.NRGBDMultiview_DIR = NRGBDMultiview_DIR
        self.sequence_list = seq_list
        if self.split == 'teaser':
            self.sequence_list = ['morning_apartment']
            self.sequence_list = ['grey_white_room']
        self.sequence_list_len = len(self.sequence_list)

        logging.info(f"NRGBDMultiview_DIR is {self.NRGBDMultiview_DIR}")
        logging.info(f"{self.status}: NRGBDMultiview Real Data size: {self.sequence_list_len}")
        logging.info(f"{self.status}: NRGBDMultiview Data dataset length: {len(self)}")

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if 'nan' in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
                poses.append(pose_floats)

        return np.array(poses, dtype=np.float32), valid
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
        seq_name = self.sequence_list[seq_index]
        seq_path = os.path.join(self.NRGBDMultiview_DIR, seq_name)
        available_images = natsorted(glob.glob(osp.join(seq_path, "images", "*.png")))
        num_images = len(available_images)
        # print(num_images)
        ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=16) # nrgbd has really close timestamps
        if self.split == 'teaser':
            ids = np.array([201, 615, 648, 12, 239, 586, 276, 298])
            ids = np.array([809, 850, 780, 741, 726, 879, 826, 919])
            ids = np.array([1204, 1252, 1307, 1328, 1356, 1280, ])
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


        # extrinsics
        posepath = osp.join(self.NRGBDMultiview_DIR, seq_name, f'poses.txt')
        camera_poses, valids = self.load_poses(posepath)
        for view_index in ids:

            # intrinsics
            fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
            intri_opencv = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            impath = osp.join(self.NRGBDMultiview_DIR, seq_path, 'images', f'img{view_index}.png')
            
            # image
            image = imread_cv2(impath)

            # extrinsics (opencv format)
            camera_pose = camera_poses[int(view_index)]
            camera_pose[:, 1:3] *= -1.0
            extri_opencv = closed_form_inverse_se3(camera_pose[None])[0][:3, :] # w2c
            
            # depth map
            depthpath = osp.join(self.NRGBDMultiview_DIR, seq_path, 'depth', f'depth{view_index}.png')
            depth_map = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depth_map = np.nan_to_num(depth_map.astype(np.float32), 0.0) / 1000.0
            depth_map[depth_map>10] = 0
            depth_map[depth_map<1e-3] = 0

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

        set_name = "NRGBDMultiview"
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


if __name__ == "__main__":
    dataset = NRGBDMultiviewDataset()
    print(dataset[10])