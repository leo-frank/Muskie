import os

import cv2
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import transforms


class ScanNetSequenceDataset(Dataset):
    """
    ScanNet loader for multi-view tracking sequences.
    """

    def __init__(
        self,
        root_dir="/home/pdl/liusidun_3d/lwy_3d/scannet_test_1500",
        num_views=8,
        image_size=(480, 640),
        max_frame_skip=35,
        split="valid",
    ):
        super().__init__()

        print("Initializing ScanNetSequenceDataset...")
        self.root_dir = root_dir
        self.num_views = num_views
        self.image_size = image_size
        self.max_frame_skip = max_frame_skip

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.intrinsics_dict = {
            k: torch.from_numpy(v).float()
            for k, v in np.load(os.path.join(root_dir, "intrinsics.npz")).items()
        }

        self.sequences = self._build_sequences()

        print(
            f"ScanNetSequenceDataset | Found {len(self.sequences)} valid sequences of length {self.num_views}."
        )

    def _build_sequences(self):
        sequences = []
        scene_ids = sorted(
            [
                d
                for d in os.listdir(self.root_dir)
                if d.startswith("scene")
                and os.path.isdir(os.path.join(self.root_dir, d))
            ]
        )

        for scene_id in scene_ids:
            pose_dir = os.path.join(self.root_dir, scene_id, "pose")
            if not os.path.exists(pose_dir):
                continue

            frame_files = natsorted(os.listdir(pose_dir))
            frame_ids = [os.path.splitext(f)[0] for f in frame_files]
            frame_numbers = [int(fid) for fid in frame_ids]

            if len(frame_numbers) < self.num_views:
                continue

            for i in range(len(frame_numbers) - self.num_views + 1):
                is_valid_sequence = True
                for j in range(self.num_views - 1):
                    frame_diff = frame_numbers[i + j + 1] - frame_numbers[i + j]
                    if frame_diff > self.max_frame_skip:
                        is_valid_sequence = False
                        break

                if is_valid_sequence:
                    sequence_frame_ids = frame_ids[i : i + self.num_views]
                    sequences.append(
                        {
                            "scene_id": scene_id,
                            "frame_ids": sequence_frame_ids,
                            "intrinsics": self.intrinsics_dict[scene_id],
                        }
                    )
                    break

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence_info = self.sequences[index]
        scene_id = sequence_info["scene_id"]
        frame_ids = sequence_info["frame_ids"]
        intrinsics_tensor = sequence_info["intrinsics"]

        images, depths, poses, masks = [], [], [], []

        for frame_id in frame_ids:
            img_path = os.path.join(self.root_dir, scene_id, "color", f"{frame_id}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            img = cv2.resize(
                img,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(img_rgb)
            images.append(img_tensor)

            depth_path = os.path.join(
                self.root_dir, scene_id, "depth", f"{frame_id}.png"
            )
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Depth not found at {depth_path}")
            depth = cv2.resize(
                depth,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            depth_tensor = torch.from_numpy(
                depth.astype(np.float32) / 1000.0
            ).unsqueeze(0)
            depths.append(depth_tensor)

            pose_path = os.path.join(self.root_dir, scene_id, "pose", f"{frame_id}.txt")
            pose = torch.from_numpy(np.loadtxt(pose_path)).float()
            Rt = pose.inverse()  # ScanNet pose files are c2w; convert to w2c.
            poses.append(Rt)

            masks.append((depth_tensor > 0).float())

        return {
            "image": torch.stack(images),
            "depth": torch.stack(depths),
            "intrinsics": intrinsics_tensor,
            "Rt": torch.stack(poses),
            "mask": torch.stack(masks),
        }
