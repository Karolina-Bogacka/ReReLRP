import os
import random
from collections import defaultdict

import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

dir_structure_help = " "


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip(
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip", root_dir
            )
        train_path = os.path.join(root_dir, "train")
        val_path = os.path.join(root_dir, "val")
        test_path = os.path.join(root_dir, "test")

        wnids_path = os.path.join(root_dir, "wnids.txt")
        words_path = os.path.join(root_dir, "words.txt")

        self._make_paths(train_path, val_path, test_path, wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path, wnids_path, words_path):
        self.ids = []
        with open(wnids_path, "r") as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, "r") as wf:
            for line in wf:
                nid, labels = line.split("\t")
                labels = list(map(lambda x: x.strip(), labels.split(",")))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            "train": [],  # [img_path, id, nid, box]
            "val": [],  # [img_path, id, nid, box]
            "test": [],  # img_path
        }

        # Get the test paths
        self.paths["test"] = list(
            map(lambda x: os.path.join(test_path, x), os.listdir(test_path))
        )
        # Get the validation paths and labels
        with open(os.path.join(val_path, "val_annotations.txt")) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, "images", fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths["val"].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + "_boxes.txt")
            imgs_path = os.path.join(train_path, nid, "images")
            label_id = self.ids.index(nid)
            with open(anno_path, "r") as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths["train"].append((fname, label_id, nid, bbox))


"""Datastructure for the tiny image dataset.
Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""


class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        preload=True,
        load_transform=None,
        transform=None,
        download=False,
        max_samples=None,
    ):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[: self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros(
                (self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32
            )
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != "test":
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = None if self.mode == "test" else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            lbl = None if self.mode == "test" else s[self.label_idx]
        sample = {"image": img, "label": lbl}

        if self.transform:
            sample = self.transform(sample)
        return sample


class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data["y"]
        self.images = data["x"]
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.open(self.images[index]).convert("RGB")
        x = self.transform(x)
        y = self.labels[index]
        return x, y


def get_data(
    path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None
):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []

    # read filenames and labels
    trn_lines = np.loadtxt(os.path.join(path, "train_loc.txt"), dtype=str)
    tst_lines = np.loadtxt(os.path.join(path, "test.txt"), dtype=str)
    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (
            num_tasks - 1
        ), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array(
            [nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1)
        )
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert (
        num_classes == cpertask.sum()
    ), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]["name"] = "task-" + str(tt)
        data[tt]["trn"] = {"x": [], "y": []}
        data[tt]["val"] = {"x": [], "y": []}
        data[tt]["tst"] = {"x": [], "y": []}

    # ALL OR TRAIN
    for this_image, this_label in trn_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]["trn"]["x"].append(this_image)
        data[this_task]["trn"]["y"].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in tst_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]["tst"]["x"].append(this_image)
        data[this_task]["tst"]["y"].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]["ncla"] = len(np.unique(data[tt]["trn"]["y"]))
        assert (
            data[tt]["ncla"] == cpertask[tt]
        ), "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]["ncla"]):
                cls_idx = list(np.where(np.asarray(data[tt]["trn"]["y"]) == cc)[0])
                rnd_img = random.sample(
                    cls_idx, int(np.round(len(cls_idx) * validation))
                )
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]["val"]["x"].append(data[tt]["trn"]["x"][rnd_img[ii]])
                    data[tt]["val"]["y"].append(data[tt]["trn"]["y"][rnd_img[ii]])
                    data[tt]["trn"]["x"].pop(rnd_img[ii])
                    data[tt]["trn"]["y"].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]["ncla"]))
        n += data[t]["ncla"]
    data["ncla"] = n

    return data, taskcla, class_order
