import random
import glob
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tvF
from natsort import natsorted


def get_segment(filenames, n):
    filenames = natsorted(filenames)
    range_length = min(len(filenames), n)
    start_idx = random.randint(0, len(filenames) - range_length)
    return filenames[start_idx:start_idx+range_length]

DATA_PATH = '/Data'

class Arkitscenes:
    def __init__(self):
        self.root = f'{DATA_PATH}/arkitscenes_processed/Training'

    def get_scene_list(self):
        root = self.root
        scene_list = [scene for scene in os.listdir(root) if os.path.isdir(os.path.join(root, scene))]
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return get_segment(glob.glob(os.path.join(root, scene, 'vga_wide/*.jpg')), 4*8)

class Scannetpp:
    def __init__(self):
        self.root = f'{DATA_PATH}/scannetpp_processed'

    def get_scene_list(self):
        root = self.root
        scene_list = [scene for scene in os.listdir(root) if not scene in ('scannetpp_pairs', 'all_metadata.npz')]
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return get_segment(glob.glob(os.path.join(root, scene, 'images/*.jpg')), 4*8)

class Co3d:
    def __init__(self):
        self.root = f'{DATA_PATH}/dust3r_datasets/co3d_processed'

    def get_scene_list(self):
        root = self.root
        scene_list = glob.glob(os.path.join(root, '**/[0-9]*_*_*'))
        scene_list = [os.path.relpath(scene, root) for scene in scene_list]
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return get_segment(glob.glob(os.path.join(root, scene, 'images/*.jpg')), 4*8)

    def load_image(self, filename, zoom_in):
        image = Image.open(filename)
        if zoom_in:
            mask = Image.open(filename.replace("images", "masks").replace(".jpg", ".png"))
            ys, xs = np.where(np.asarray(mask) > 0)
            if len(xs) == 0 or len(ys) == 0:
                return image
            xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
            w, h = xmax - xmin + 1, ymax - ymin + 1
            W, H = mask.size
            xmin = max(0, xmin - w * 0.5)
            xmax = min(W - 1, xmax + w * 0.5)
            ymin = max(0, ymin - h * 0.5)
            ymax = min(H - 1, ymax + h * 0.5)
            image = image.crop((xmin, ymin, xmax, ymax))
        return image

class BlendedMVG:
    def __init__(self):
        self.root = f'{DATA_PATH}/blendedmvg_uncompressed/rgb/blended_mvg'

    def get_scene_list(self):
        root = self.root
        scene_list = [scene for scene in os.listdir(root) if len(os.listdir(os.path.join(root, scene))) > 0]
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return get_segment(glob.glob(os.path.join(root, scene, '*.png')), 4*8)

class HyperSim:
    def __init__(self):
        self.root = f'{DATA_PATH}/hypersim/downloads'

    def get_scene_list(self):
        root = self.root
        scene_list = [scene for scene in os.listdir(root) if not scene in ('metadata_camera_parameters.csv',)]
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return get_segment(glob.glob(os.path.join(root, scene, 'images/scene_cam_*_final_preview/*.jpg')), 2*8)

class Waymo:
    def __init__(self):
        self.root = f'{DATA_PATH}/waymo_processed'

    def get_scene_list(self):
        root = self.root
        scene_list = [scene for scene in os.listdir(root) if not scene in ('waymo_exist_pairs.npz',) and len(os.listdir(os.path.join(root, scene))) > 0]
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return glob.glob(os.path.join(root, scene, '*.jpg'))

class MegaDepth:
    def __init__(self):
        self.root = f'{DATA_PATH}/megadepth_processed'
        data = np.load(os.path.join(self.root, 'all_metadata_8view.npz'), allow_pickle=True)
        self.pairs, self.scenes, self.images = data['pairs'], data['scenes'], data['images']

    def get_scene_list(self):
        return self.pairs

    def get_filenames(self, pair):
        scene_id, group = pair
        scene = self.scenes[scene_id]
        filenames = [os.path.join(self.root, *scene.split(), self.images[i]+'.jpg') for i in group]
        return filenames

class DL3DV:
    def __init__(self):
        self.root = f'{DATA_PATH}/DL3DV-10K'

    def get_scene_list(self):
        root = self.root
        scene_list = []
        for scene in glob.glob(os.path.join(root, '*K/**')):
            if len(os.listdir(os.path.join(scene, 'images_8'))) > 0:
                scene_list.append(os.path.relpath(scene, root))
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return get_segment(glob.glob(os.path.join(root, scene, 'images_8/*.png')), 8*8)

class RE10K:
    def __init__(self):
        self.root = f'{DATA_PATH}/re10k/test/images'

    def get_scene_list(self):
        root = self.root
        scene_list = os.listdir(root)
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return get_segment(glob.glob(os.path.join(root, scene, '*.jpg')), 4*8)

class WildRGBD:
    def __init__(self):
        self.root = f'{DATA_PATH}/wildrgbd/subset'

    def get_scene_list(self):
        root = self.root
        scene_list = glob.glob(os.path.join(root, '**/scenes/scene_*'))
        scene_list = [os.path.relpath(scene, root) for scene in scene_list]
        return scene_list

    def get_filenames(self, scene):
        root = self.root
        return get_segment(glob.glob(os.path.join(root, scene, 'rgb/*.png')), 4*8)

    def load_image(self, filename, zoom_in):
        image = Image.open(filename)
        if zoom_in:
            mask = Image.open(filename.replace("rgb/", "masks/"))
            ys, xs = np.where(np.asarray(mask) > 0)
            if len(xs) == 0 or len(ys) == 0:
                return image
            xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
            w, h = xmax - xmin + 1, ymax - ymin + 1
            W, H = mask.size
            xmin = max(0, xmin - w * 0.5)
            xmax = min(W - 1, xmax + w * 0.5)
            ymin = max(0, ymin - h * 0.5)
            ymax = min(H - 1, ymax + h * 0.5)
            image = image.crop((xmin, ymin, xmax, ymax))
        return image


class MultiviewDataset(Dataset):
    def __init__(self, num_views, image_size=224, epoch_size=100_000):
        self.num_views = num_views
        self.epoch_size = epoch_size
        self.image_size = image_size
        self.datasets = [Co3d(), Arkitscenes(), BlendedMVG(), Scannetpp(), DL3DV(), HyperSim(), RE10K(), WildRGBD()]
        if num_views == 8:
            self.datasets.append(MegaDepth())
        self.scene_list = [l.get_scene_list() for l in self.datasets]
        self.transform = None
        print("Dataset spec:")
        for dataset, scene_list in zip(self.datasets, self.scene_list):
            print(f"\t{dataset.__class__.__name__}: {len(scene_list)} scenes;")

    def __len__(self):
        return self.epoch_size

    def build_transform(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), interpolation=3), # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def __getitem__(self, index):
        if self.transform is None:
            self.build_transform()
        dataset_idx = random.randint(0, len(self.datasets)-1)
        dataset = self.datasets[dataset_idx]
        scene_idx = random.randint(0, len(self.scene_list[dataset_idx])-1)
        scene = self.scene_list[dataset_idx][scene_idx]
        filenames = dataset.get_filenames(scene)
        import copy
        remain_filenames = copy.copy(filenames)
        images = []
        if isinstance(dataset, (Co3d, WildRGBD)):
            zoom_in = np.random.random() > 0.5
        else:
            zoom_in = False
        while len(images) < self.num_views:
            filename = random.choice(remain_filenames)
            remain_filenames.remove(filename)
            if len(remain_filenames) == 0:
                remain_filenames = copy.copy(filenames)
            try:
                if zoom_in:
                    images.append(self.transform(self.datasets[dataset_idx].load_image(filename, zoom_in=zoom_in)))
                else:
                    images.append(self.transform(Image.open(filename)))
            except OSError:
                print("Image broken:", filename)
            except ZeroDivisionError:
                print("Mask broken:", filename)
        images = torch.stack(images)
        return images, dataset.__class__.__name__


class MultiresWrapper(Dataset):
    def __init__(self, num_views, image_size_list=[224, 384, 512], epoch_size=100_000):
        dataset = MultiviewDataset(num_views, image_size_list[0], epoch_size)
        import copy
        self.datasets = dict()
        for image_size in image_size_list:
            dataset_copy = copy.deepcopy(dataset)
            dataset_copy.image_size = image_size
            self.datasets[image_size] =  dataset_copy
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        all_images = {}
        for res, dataset in self.datasets.items():
            all_images[res] = dataset[index]
        return all_images
