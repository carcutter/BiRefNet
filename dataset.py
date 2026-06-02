import os
import csv
import random
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from image_proc import preproc
from config import Config
from utils import path_to_image, path_to_binary_mask_from_colors, path_to_binary_mask_nonbg


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()
_class_labels_TR_sorted = (
    'Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, '
    'BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, '
    'CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, '
    'Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, '
    'Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, '
    'Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, '
    'KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, '
    'Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, '
    'OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, '
    'RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, '
    'ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, '
    'Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, '
    'TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, '
    'UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht'
)
class_labels_TR_sorted = _class_labels_TR_sorted.split(', ')


class MyData(data.Dataset):
    def __init__(self, datasets, data_size, is_train=True, csv_path=None, csv_image_root=None,
                 max_samples=0, val_split=0.0, csv_split_seed=42):
        # data_size is None when using dynamic_size or data_size is manually set to None (for inference in the original size).
        # csv_path: if set, load (image_path, mask_path) pairs from a CSV with those columns.
        #           Paths inside the CSV resolve against csv_image_root (defaults to the CSV's
        #           own directory). `datasets` is ignored in this mode.
        # val_split > 0: treat csv_path as a full index and produce the train (1-val_split)
        #           or val (val_split) slice. The split is deterministic w.r.t. csv_split_seed
        #           — calling this constructor twice with the same seed but flipped is_train
        #           yields disjoint, complementary subsets.
        self.is_train = is_train
        self.data_size = data_size
        self.load_all = config.load_all
        self.device = config.device
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']

        if self.is_train and config.auxiliary_classification:
            self.cls_name2id = {_name: _id for _id, _name in enumerate(class_labels_TR_sorted)}
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
        ])

        if csv_path is not None:
            # Different CSVs use different conventions for relative paths: index.csv stores them
            # relative to csv_image_root (data_link), while data/processed/*.csv (e.g. blob_crops.csv)
            # store them relative to the repo root. Try each candidate root and use the first where
            # the file actually exists, so both layouts load without per-file config.
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            candidate_roots = [r for r in (csv_image_root, os.getcwd(), csv_dir) if r]

            def _resolve(rel):
                if os.path.isabs(rel):
                    return rel
                for root in candidate_roots:
                    cand = os.path.join(root, rel)
                    if os.path.exists(cand):
                        return cand
                return os.path.join(candidate_roots[0], rel)   # let a clear FileNotFound surface downstream

            pairs = []   # list of (img_abs, msk_abs)
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Accept both schemas: image_path/mask_path (index.csv) and image/mask (blob_crops.csv).
                    img_rel = (row.get('image_path') or row.get('image') or '').strip()
                    msk_rel = (row.get('mask_path') or row.get('mask') or '').strip()
                    if not img_rel or not msk_rel:
                        continue
                    pairs.append((_resolve(img_rel), _resolve(msk_rel)))

            if val_split and 0.0 < val_split < 1.0:
                # Deterministic 80/20-style split. Same seed + flipped is_train ⇒ disjoint subsets.
                rng = np.random.default_rng(csv_split_seed)
                idx = np.arange(len(pairs))
                rng.shuffle(idx)
                n_val = int(round(len(pairs) * val_split))
                val_idx = set(idx[:n_val].tolist())
                keep = (lambda i: i not in val_idx) if is_train else (lambda i: i in val_idx)
                pairs = [p for i, p in enumerate(pairs) if keep(i)]

            self.image_paths = [p[0] for p in pairs]
            self.label_paths = [p[1] for p in pairs]
            if max_samples and len(self.image_paths) > max_samples:
                self.image_paths = self.image_paths[:max_samples]
                self.label_paths = self.label_paths[:max_samples]
        else:
            dataset_root = os.path.join(config.data_root_dir, config.task)
            # datasets can be a list of different datasets for training on combined sets.
            self.image_paths = []
            for dataset in datasets.split('+'):
                image_root = os.path.join(dataset_root, dataset, 'im')
                self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root) if any(p.endswith(ext) for ext in valid_extensions)]
            self.label_paths = []
            for p in self.image_paths:
                for ext in valid_extensions:
                    ## 'im' and 'gt' may need modifying
                    p_gt = p.replace('/im/', '/gt/')[:-(len(p.split('.')[-1])+1)] + ext
                    file_exists = False
                    if os.path.exists(p_gt):
                        self.label_paths.append(p_gt)
                        file_exists = True
                        break
                if not file_exists:
                    print('Not exists:', p_gt)

        if len(self.label_paths) != len(self.image_paths):
            set_image_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths])
            set_label_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths])
            print('Path diff:', set_image_paths - set_label_paths)
            raise ValueError(f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})")

        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=self.data_size, color_type='rgb')
                _label = self._load_label(label_path)
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1
                )

    def _load_label(self, path):
        mode = getattr(config, 'mask_color_to_binary', False)
        # Back-compat: True ⇒ legacy 'colors' mode; False / falsy ⇒ grayscale luminance.
        if mode is True:
            mode = 'colors'
        elif mode is False or mode is None or mode == '':
            mode = 'none'
        if mode == 'nonbg':
            return path_to_binary_mask_nonbg(path, size=self.data_size, bg_colors=config.mask_bg_colors)
        if mode == 'colors':
            return path_to_binary_mask_from_colors(path, size=self.data_size, fg_colors=config.mask_fg_colors)
        return path_to_image(path, size=self.data_size, color_type='gray')

    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = self.class_labels_loaded[index] if self.is_train and config.auxiliary_classification else -1
        else:
            image = path_to_image(self.image_paths[index], size=self.data_size, color_type='rgb')
            label = self._load_label(self.label_paths[index])
            class_label = self.cls_name2id[self.label_paths[index].split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1

        # loading image and label
        if self.is_train:
            if config.background_color_synthesis:
                image.putalpha(label)
                array_image = np.array(image)
                array_foreground = array_image[:, :, :3].astype(np.float32)
                array_mask = (array_image[:, :, 3:] / 255).astype(np.float32)
                array_background = np.zeros_like(array_foreground)
                choice = random.random()
                if choice < 0.4:
                    # Black/Gray/White backgrounds
                    array_background[:, :, :] = random.randint(0, 255)
                elif choice < 0.8:
                    # Background color that similar to the foreground object. Hard negative samples.
                    foreground_pixel_number = np.sum(array_mask > 0)
                    color_foreground_mean = np.mean(array_foreground * array_mask, axis=(0, 1)) * (np.prod(array_foreground.shape[:2]) / foreground_pixel_number)
                    color_up_or_down = random.choice((-1, 1))
                    # Up or down for 20% range from 255 or 0, respectively.
                    color_foreground_mean += (255 - color_foreground_mean if color_up_or_down == 1 else color_foreground_mean) * (random.random() * 0.2) * color_up_or_down
                    array_background[:, :, :] = color_foreground_mean
                else:
                    # Any color
                    for idx_channel in range(3):
                        array_background[:, :, idx_channel] = random.randint(0, 255)
                array_foreground_background = array_foreground * array_mask + array_background * (1 - array_mask)
                image = Image.fromarray(array_foreground_background.astype(np.uint8))
            image, label = preproc(image, label, preproc_methods=config.preproc_methods)
        # else:
        #     if _label.shape[0] > 2048 or _label.shape[1] > 2048:
        #         _image = cv2.resize(_image, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        #         _label = cv2.resize(_label, (2048, 2048), interpolation=cv2.INTER_LINEAR)

        # At present, we use fixed sizes in inference, instead of consistent dynamic size with training.
        if self.is_train:
            if config.dynamic_size is None:
                image, label = self.transform_image(image), self.transform_label(label)
        else:
            size_div_32 = (int(image.size[0] // 32 * 32), int(image.size[1] // 32 * 32))
            if image.size != size_div_32:
                image = image.resize(size_div_32)
                label = label.resize(size_div_32)
            image, label = self.transform_image(image), self.transform_label(label)

        if self.is_train:
            return image, label, class_label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)


def custom_collate_fn(batch):
    if config.dynamic_size:
        dynamic_size = tuple(sorted(config.dynamic_size))
        dynamic_size_batch = (random.randint(dynamic_size[0][0], dynamic_size[0][1]) // 32 * 32, random.randint(dynamic_size[1][0], dynamic_size[1][1]) // 32 * 32) # select a value randomly in the range of [dynamic_size[0/1][0], dynamic_size[0/1][1]].
        data_size = dynamic_size_batch
    else:
        data_size = config.size
    new_batch = []
    transform_image = transforms.Compose([
        transforms.Resize(data_size[::-1]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_label = transforms.Compose([
        transforms.Resize(data_size[::-1]),
        transforms.ToTensor(),
    ])
    for image, label, class_label in batch:
        new_batch.append((transform_image(image), transform_label(label), class_label))
    return data._utils.collate.default_collate(new_batch)
