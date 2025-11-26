# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
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


Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
def read_images_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images
def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


seq_list = [

]

class ETH3DMultiviewDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        ETH3DMultiview_DIR: str = "/media/pdl631/lxd_run/liwenyu_3d/dust3r_datasets/eth3d/multi_view_test_dslr_undistorted",
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the ETH3DMultiviewDataset.
        """
        super(ETH3DMultiviewDataset, self).__init__(len_train, len_test, split, expand_ratio)

        self.ETH3DMultiview_DIR = ETH3DMultiview_DIR
        self.sequence_list = sorted(os.listdir(osp.join(self.ETH3DMultiview_DIR)))
        self.sequence_list_len = len(self.sequence_list)

        logging.info(f"ETH3DMultiview_DIR is {self.ETH3DMultiview_DIR}")
        logging.info(f"{self.status}: ETH3DMultiview Real Data size: {self.sequence_list_len}")
        logging.info(f"{self.status}: ETH3DMultiview Data dataset length: {len(self)}")

    def _build_intrinsics(self, cameras_all):
        intrinsics_all = {}
        for cam_id, cam in cameras_all.items():
            model, param = cam.model, cam.params
            if model == 'PINHOLE':
                fx, fy, cx, cy = param
            elif model == 'SIMPLE_PINHOLE':
                fx, cx, cy = param
                fy = fx
            else:
                assert False
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            intrinsics_all[cam_id] = K
        return intrinsics_all
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

        image_path = osp.join(self.ROOT, seq_name, 'images')
        cam_path = osp.join(self.ROOT, seq_name, 'dslr_calibration_undistorted')
        intrinsics_all = self._build_intrinsics(read_cameras_text(osp.join(cam_path, 'cameras.txt')))
        images_all = read_images_text(osp.join(cam_path, 'images.txt'))

        img_idxs = sorted(images_all.keys())
        num_images = len(img_idxs)
        ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)
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

            image_info = images_all[view_index]
            imname = image_info.name
            impath = osp.join(image_path, imname)
            
            # image
            rgb_image = imread_cv2(impath)

            rotation = image_info.qvec2rotmat()
            translation = image_info.tvec
            camera_pose = np.eye(4, dtype=np.float32) # w2c
            camera_pose[:3, :3] = rotation
            camera_pose[:3, 3] = translation
            extri_opencv = camera_pose # w2c

            intrinsics = intrinsics_all[image_info.camera_id]

            # intrinscis and extrinsics (opencv format)
            camera_params = np.load(impath.replace("jpg", "npz"))
            intri_opencv = np.float32(camera_params['intrinsics'])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params['R_cam2world']
            camera_pose[:3, 3] = camera_params['t_cam2world']
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

        set_name = "ETH3DMultiview"
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
    dataset = ETH3DMultiviewDataset()
    print(dataset[0])