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
import torch
import cv2
import numpy as np
from downstreams.utils.geometry import closed_form_inverse_se3
from torch.utils.data import Dataset
from downstreams.datasets.dataset_util import *
from downstreams.datasets.base_dataset import BaseDataset




class WaymoDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        Waymo_DIR: str = "/media/pdl631/lxd_run/liwenyu_3d/dust3r_datasets/waymo_processed/",
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the WaymoDataset.
        """
        super(WaymoDataset, self).__init__(len_train, len_test, split, expand_ratio)

        self.Waymo_DIR = Waymo_DIR
        self.sequence_list = [i for i in os.listdir(self.Waymo_DIR) if 'waymo_exist_pairs' not in i ]
        self.sequence_list_len = len(self.sequence_list)

        logging.info(f"Waymo_DIR is {self.Waymo_DIR}")
        logging.info(f"{self.status}: Waymo Real Data size: {self.sequence_list_len}")
        logging.info(f"{self.status}: Waymo Data dataset length: {len(self)}")

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
        seq_path = os.path.join(self.Waymo_DIR, seq_name)
        available_images = glob.glob(osp.join(seq_path, "*.jpg"))
        num_images = len(available_images)
        if num_images == 0:
            print(seq_name)
        # print(num_images)
        ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)
        # print(ids)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

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

        for view_index in ids:

            impath = available_images[view_index]
            
            # image
            image = imread_cv2(impath)

            # intrinscis and extrinsics (opencv format)
            camera_params = np.load(impath.replace("jpg", "npz"))
            intri_opencv = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])
            extri_opencv = closed_form_inverse_se3(camera_pose[None])[0][:3, :] # w2c
            
            # depth map
            depth_map = imread_cv2(impath.replace("jpg", "exr"))

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

        set_name = "Waymo"
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
    dataset = WaymoDataset()
    print(dataset[0])