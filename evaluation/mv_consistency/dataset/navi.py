import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from .utils import (
    bbox_crop,
    camera_matrices_from_annotation,
    compute_normal,
    get_grid,
    get_navi_transforms,
    read_depth,
    read_image,
)


class NAVI(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split="train",
        model="all",
        image_mean="None",
        rotateflip=False,
        relative_depth=False,
        num_views=8,
        reorder=False,
    ):
        super().__init__()

        # generate splits based on collections and subparts
        if split == "train":
            collection = "multiview"
            subpart = "train"
        elif split == "valid":
            collection = "multiview"
            subpart = "test"
        elif split == "trainval":  # here
            collection = "multiview"
            subpart = "all"
        elif split == "test":
            collection = "wild"
            subpart = "all"
        else:
            raise ValueError(f"Unknown split: {split}")

        # set path
        self.data_root = Path(path)
        self.bbox_crop = True
        self.relative_depth = False
        self.max_depth = 1.0
        self.reorder = reorder

        self.name = f"NAVI_{collection}_{subpart}"
        if relative_depth:
            self.name = self.name + "_reldepth"

        # parse dataset
        self.data_dict = self.parse_dataset()
        self.define_instances_split(model, collection, subpart)

        # get transforms
        augment = False
        image_size = (512, 512)
        t_fns = get_navi_transforms(
            image_mean,
            image_size=image_size,
            augment=augment,
            rotateflip=rotateflip,
            additional_targets={
                "depth": "image",
                "snorm": "image",
                "xyz_grid": "image",
            },
        )
        self.image_transform, self.target_transform, self.shared_transform = t_fns
        print(f"NAVI {collection} {subpart} {model}: {len(self.instances)} instances")

        self.num_views = num_views
        self.instance_clusters = self.generate_instance_clusters(
            self.instances, self.num_views
        )[::20]

    def __len__(self):
        return len(self.instance_clusters)

    def __getitem__(self, index):
        cluster_info = self.instance_clusters[index]

        all_data = []
        for obj_id, scene_id, img_id in cluster_info:
            single_instance_data = self.get_single(obj_id, scene_id, img_id)
            all_data.append(single_instance_data)

        batch = {k: [d[k] for d in all_data] for k in all_data[0]}

        for key in batch:
            if isinstance(batch[key][0], torch.Tensor):
                batch[key] = torch.stack(batch[key], dim=0)

        return batch

    def generate_instance_clusters(self, instances, num_views):
        """
        Build view clusters for each instance.
        """
        inst_dict = {}
        for obj_id, coll_id, img_id in instances:
            if obj_id not in inst_dict:
                inst_dict[obj_id] = {coll_id: [img_id]}
            elif coll_id not in inst_dict[obj_id]:
                inst_dict[obj_id][coll_id] = [img_id]
            else:
                inst_dict[obj_id][coll_id].append(img_id)

        clusters = []
        for obj_id in inst_dict:
            for col_id in inst_dict[obj_id]:
                scene_img_ids = inst_dict[obj_id][col_id]

                if len(scene_img_ids) < num_views:
                    continue
                rots = []
                for img_id in scene_img_ids:
                    anno = self.data_dict[obj_id][col_id]["annotations"][img_id]
                    Rt = camera_matrices_from_annotation(anno)
                    rots.append(Rt[:3, :3])
                rots = torch.stack(rots, dim=0)

                for i, anchor_img_id in enumerate(scene_img_ids):
                    rot_i = rots[i, None].repeat(len(rots), 1, 1)
                    rots_ij = torch.bmm(rot_i, rots.permute(0, 2, 1))
                    rots_tr = torch.einsum("bii->b", rots_ij)
                    rel_ang_rad = (0.5 * rots_tr - 0.5).clamp(min=-1, max=1).acos()

                    _, nearest_indices = torch.topk(
                        rel_ang_rad, k=num_views, largest=False
                    )

                    cluster_img_ids = [scene_img_ids[j] for j in nearest_indices]
                    if self.reorder:
                        tmp = cluster_img_ids[0]
                        cluster_img_ids[0] = cluster_img_ids[-1]
                        cluster_img_ids[-1] = tmp
                    cluster_info = [
                        (obj_id, col_id, img_id) for img_id in cluster_img_ids
                    ]
                    clusters.append(cluster_info)

        return clusters

    def get_single(self, obj_id, scene_id, img_id):
        obj_number = self.objects[obj_id]
        anno = self.data_dict[obj_id][scene_id]["annotations"][img_id]

        # get scene path
        prefix = "downsampled_"
        scene_path = self.data_root / obj_id / scene_id
        image_path = scene_path / f"images/{prefix}{img_id}.jpg"
        depth_path = scene_path / f"depth/{prefix}{img_id}.png"

        # get image
        image = read_image(image_path)
        image = self.image_transform(image)

        # get depth -- move from millimeter to meters
        depth = read_depth(str(depth_path)) / 1000
        min_depth = depth[depth > 0].min()
        depth = self.target_transform(depth)

        orig_h, orig_w = anno["image_size"]
        image_h, image_w = image.shape[1:]
        orig_fx = anno["camera"]["focal_length"]
        aug_fx = orig_fx * min(image_h, image_w) / min(orig_h, orig_w)

        intrinsics = torch.eye(3)
        intrinsics[0, 0] = aug_fx
        intrinsics[1, 1] = aug_fx
        intrinsics[0, 2] = image_w / 2.0
        intrinsics[1, 2] = image_h / 2.0

        if self.bbox_crop:
            image, depth, intrinsics_after_crop = bbox_crop(image, depth, intrinsics)
        else:
            intrinsics_after_crop = intrinsics

        bbox_h, bbox_w = image.shape[1:]

        final_h, final_w = 512, 512
        scale_w = final_w / bbox_w
        scale_h = final_h / bbox_h

        intrinsics_final = intrinsics_after_crop.clone()
        intrinsics_final[0, 0] *= scale_w
        intrinsics_final[1, 1] *= scale_h
        intrinsics_final[0, 2] *= scale_w
        intrinsics_final[1, 2] *= scale_h

        snorm = compute_normal(depth.clone(), intrinsics_after_crop[0, 0])

        if self.shared_transform is not None:
            transformed = self.shared_transform(
                image=image.permute(1, 2, 0).numpy(),
                depth=depth.permute(1, 2, 0).numpy(),
                snorm=snorm.permute(1, 2, 0).numpy(),
            )

            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
            snorm = torch.tensor(transformed["snorm"]).float().permute(2, 0, 1)
        else:
            assert (bbox_h, bbox_w) == (final_h, final_w), (
                "Mismatch size without transform"
            )

        depth[depth < min_depth] = 0

        grid = get_grid(final_h, final_w)
        uvd_grid = depth * grid
        xyz = intrinsics_final.inverse() @ uvd_grid.view(3, final_h * final_w)
        xyz_grid = xyz.view(3, final_h, final_w)

        intrinsics = intrinsics_final

        Rt = camera_matrices_from_annotation(anno)  # NAVI pose is w2c.
        Rt[:3, 3] = Rt[:3, 3] / 1000.0

        if self.relative_depth:
            max_depth = depth.max()
            zero_mask = depth == 0

            depth = (depth - min_depth) / max(0.01, max_depth - min_depth)
            depth = depth * 0.99 + 0.01

            depth[zero_mask] = 0

        mask = (depth > 0).float()
        return {
            "image": image,
            "depth": depth,
            "class_id": obj_number,
            "intrinsics": intrinsics,
            "snorm": snorm,
            "Rt": Rt,
            "anno": anno,
            "xyz_grid": xyz_grid,
            "mask": mask,
        }

    def parse_dataset(self):
        """
        Parses the directory for instances.
        Input: data_dict -- sturcture  <object_id>/<collection>/<instances>

        Output: all dataset instances
        """
        data_dict = {}

        # get all image folders
        all_collections = []
        all_collections += sorted(glob.glob(str(self.data_root / "*/multiview_*")))
        all_collections += sorted(glob.glob(str(self.data_root / "*/wild_set")))

        # parse image folders
        for collection_path in all_collections:
            object_id, collection_id = collection_path.split("/")[-2:]

            # get image ids
            img_files = sorted(os.listdir(os.path.join(collection_path, "images")))
            img_ids = [_file.split(".")[0] for _file in img_files if "jpg" in _file]

            # VERY hacky | remove small_
            img_ids = [_id for _id in img_ids if "_" not in _id]

            # load annotations and convert them to a dictionary
            with open(os.path.join(collection_path, "annotations.json")) as f:
                annotations = json.load(f)
                annotations = {
                    _an["filename"].split(".")[0]: _an for _an in annotations
                }

            # update data_dict
            if object_id not in data_dict:
                data_dict[object_id] = {}

            data_dict[object_id][collection_id] = {
                "views": img_ids,
                "annotations": annotations,
            }

        return data_dict

    def define_instances_split(self, model, collection, subpart):
        # get a list of object names
        if model == "all":
            object_names = list(self.data_dict.keys())
        else:
            object_names = [model]

        assert collection in ["multiview", "wild"]
        assert subpart in ["train", "test", "all"]

        self.instances = []
        self.objects = []

        for obj_id in object_names:
            scenes = sorted(self.data_dict[obj_id].keys())
            if "wild_set" not in scenes or len(scenes) == 1:
                print(f"Skipping object {obj_id}.")
                continue
            else:
                self.objects.append(obj_id)

            if collection == "wild":
                image_ids = sorted(self.data_dict[obj_id]["wild_set"]["views"])
                image_ann = self.data_dict[obj_id]["wild_set"]["annotations"]
                assert len(image_ids) > 1

                for _id in image_ids:
                    if subpart == "all":
                        self.instances.append((obj_id, "wild_set", _id))
                    else:
                        im_split = image_ann[_id]["split"]
                        if subpart == "train" and im_split == "train":
                            self.instances.append((obj_id, "wild_set", _id))
                        elif subpart == "test" and im_split == "val":
                            self.instances.append((obj_id, "wild_set", _id))
            else:
                scenes = sorted(_s for _s in scenes if "multiview" in _s)

                # int floors -> at least 1 validation scene
                train_split = int(0.9 * len(scenes))

                if subpart == "train":
                    scenes = scenes[:train_split]
                elif subpart == "test":
                    scenes = scenes[train_split:]
                elif subpart == "all":
                    scenes = scenes
                else:
                    assert collection == "multiview", f"collection was {collection}."

                if len(scenes) == 0:
                    continue

                for scene in scenes:
                    image_ids = sorted(self.data_dict[obj_id][scene]["views"])
                    for _id in image_ids:
                        self.instances.append((obj_id, scene, _id))

        # create object -> class mapping
        self.objects.sort()
        self.objects = {_obj: _id for _id, _obj in enumerate(self.objects)}

    def generate_instance_pairs(self, instances):
        torch.manual_seed(8)
        inst_dict = {}
        for ins in instances:
            obj_id, coll_id, img_id = ins
            if obj_id not in inst_dict:
                inst_dict[obj_id] = {coll_id: [img_id]}
            elif coll_id not in inst_dict[obj_id]:
                inst_dict[obj_id][coll_id] = [img_id]
            else:
                inst_dict[obj_id][coll_id].append(img_id)

        pair_dict = {}
        for obj_id in inst_dict:
            pair_dict[obj_id] = {}
            for col_id in inst_dict[obj_id]:
                pair_dict[obj_id][col_id] = {}
                rots = []
                img_ids = []
                for img_id in inst_dict[obj_id][col_id]:
                    anno = self.data_dict[obj_id][col_id]["annotations"][img_id]
                    Rt = camera_matrices_from_annotation(anno)
                    rots.append(Rt[:3, :3])
                    img_ids.append(img_id)
                rots = torch.stack(rots, dim=0)

                # for each image find a pair between 0 and max_angle degrees
                for i in range(len(img_ids)):
                    img_id = img_ids[i]
                    rots_i = rots[i, None].repeat(len(rots), 1, 1)
                    rots_ij = torch.bmm(rots_i, rots.permute(0, 2, 1))
                    rots_tr = rots_ij[:, 0, 0] + rots_ij[:, 1, 1] + rots_ij[:, 2, 2]
                    rel_ang_rad = (0.5 * rots_tr - 0.5).clamp(min=-1, max=1).acos()
                    rel_ang_deg = rel_ang_rad * 180 / np.pi

                    # only sample from deg(0, max_angle)
                    rel_ang_deg[rel_ang_deg > self.max_angle] = 0
                    rel_ang_deg[rel_ang_deg > 0] = 1

                    # sample an element
                    pair_i = torch.multinomial(rel_ang_deg, 1).item()
                    pair_dict[obj_id][col_id][img_id] = img_ids[pair_i]

        return pair_dict
