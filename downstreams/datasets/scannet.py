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




class ScannetppDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        Scannetpp_DIR: str = "",
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the ScannetppDataset.
        """
        super(ScannetppDataset, self).__init__(len_train, len_test, split, expand_ratio)

        self.Scannetpp_DIR = Scannetpp_DIR
        with np.load(osp.join(self.Scannetpp_DIR, 'all_metadata.npz')) as data:
            self.sequence_list = data['scenes']
            self.sequence_list_len = len(self.sequence_list)
            # self.intrinsics = data['intrinsics'].astype(np.float32)
            # self.trajectories = data['trajectories'].astype(np.float32)
            # self.pairs = data['pairs'][:, :2].astype(int)


        logging.info(f"Scannetpp_DIR is {self.Scannetpp_DIR}")
        logging.info(f"{self.status}: Scannetpp Real Data size: {self.sequence_list_len}")
        logging.info(f"{self.status}: Scannetpp Data dataset length: {len(self)}")

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
        seq_path = os.path.join(self.Scannetpp_DIR, seq_name, 'images')
        scene_metadata = np.load(os.path.join(self.Scannetpp_DIR, seq_name, 'scene_metadata.npz'))

        available_images = scene_metadata['images']
        trajectories = scene_metadata['trajectories'] # 4 4
        scene_intrinsics = scene_metadata['intrinsics']

        num_images = len(available_images)
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

            basename = available_images[view_index]
            
            # image
            impath = osp.join(self.Scannetpp_DIR, seq_name, 'images', basename + '.jpg')
            image = imread_cv2(impath)

            # depth
            depth_map = imread_cv2(osp.join(self.Scannetpp_DIR, seq_name, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
            depth_map = depth_map.astype(np.float32) / 1000
            depth_map[~np.isfinite(depth_map)] = 0  # invalid

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            # intrinscis and extrinsics (opencv format)
            intri_opencv = scene_intrinsics[view_index]
    
            camera_pose = np.float32(trajectories[view_index])
            extri_opencv = closed_form_inverse_se3(camera_pose[None])[0][:3, :] # w2c


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

        set_name = "Scannetpp"
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
    dataset = ScannetppDataset()
    print(dataset[0])