''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
from dataset_utils import download_file_from_google_drive, check_integrity
import pickle
import torch
import h5py as h5
import os
import os.path
import sys
import PIL
from PIL import Image
import numpy as np
from tqdm import tqdm, trange

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dogball/xxx.png
        root/dogball/xxy.png
        root/dogball/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, load_in_mem=False,
                 index_filename='imagenet_imgs.npz', **kwargs):
        classes, class_to_idx = find_classes(root)
        # Load pre-computed image directory walk
        if os.path.exists(index_filename):
            print('Loading pre-saved Index file %s...' % index_filename)
            imgs = np.load(index_filename)['imgs']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating  Index file %s...' % index_filename)
            imgs = make_dataset(root, class_to_idx)
            np.savez_compressed(index_filename, **{'imgs': imgs})
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                
                path, target = imgs[index][0], imgs[index][1]
                self.data.append(self.transform(self.loader(path)))
                self.labels.append(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]
        else:
            path, target = self.imgs[index]
            img = self.loader(str(path))
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #print(img.size(), target)
        return img, int(target)

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


''' ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
    having to load individual images all the time. '''


class ILSVRC_HDF5(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 load_in_mem=False, train=True, download=False, validate_seed=0,
                 val_split=0, **kwargs):  # last four are dummies

        self.root = root
        self.num_imgs = len(h5.File(root, 'r')['labels'])

        # self.transform = transform
        self.target_transform = target_transform

        # Set the transform here
        self.transform = transform

        # load the entire dataset into memory?
        self.load_in_mem = load_in_mem

        # If loading into memory, do so now
        if self.load_in_mem:
            print('Loading %s into memory...' % root)
            with h5.File(root, 'r') as f:
                self.data = f['imgs'][:]
                self.labels = f['labels'][:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # If loaded the entire dataset in RAM, get image from memory
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]

        # Else load it from disk
        else:
            with h5.File(self.root, 'r') as f:
                img = f['imgs'][index]
                target = f['labels'][index]

        # if self.transform is not None:
                # img = self.transform(img)
        # Apply my own transform
        img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self):
        return self.num_imgs
        # return len(self.f['imgs'])


class CIFAR10(dset.CIFAR10):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=True, validate_seed=0,
                 val_split=0, load_in_mem=True, **kwargs):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.val_split = val_split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        self.data = []
        self.labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.labels += entry['labels']
            else:
                self.labels += entry['fine_labels']
            fo.close()

        self.data = np.concatenate(self.data)
        # Randomly select indices for validation
        if self.val_split > 0:
            label_indices = [[] for _ in range(max(self.labels)+1)]
            for i, l in enumerate(self.labels):
                label_indices[l] += [i]
            label_indices = np.asarray(label_indices)

            # randomly grab 500 elements of each class
            np.random.seed(validate_seed)
            self.val_indices = []
            for l_i in label_indices:
                self.val_indices += list(l_i[np.random.choice(len(l_i), int(
                    len(self.data) * val_split) // (max(self.labels) + 1), replace=False)])

        if self.train == 'validate':
            self.data = self.data[self.val_indices]
            self.labels = list(np.asarray(self.labels)[self.val_indices])

            self.data = self.data.reshape(
                (int(50e3 * self.val_split), 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        elif self.train:
            print(np.shape(self.data))
            if self.val_split > 0:
                self.data = np.delete(self.data, self.val_indices, axis=0)
                self.labels = list(np.delete(np.asarray(
                    self.labels), self.val_indices, axis=0))

            self.data = self.data.reshape(
                (int(50e3 * (1.-self.val_split)), 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.data = self.data.reshape((10000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10Unsupervised(CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = 0
        return img, target

class CINIC10Unsupervised(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = 0
        return img, target

class Art(ImageFolder):
    def __getitem__(self, index) :
        img, target = super().__getitem__(index)
        return img, target

class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


class CIFAR100Unsupervised(CIFAR100):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = 0
        return img, target
class STL10(dset.STL10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=True, validate_seed=0,
                 val_split=0, load_in_mem=True, **kwargs):
        super(STL10, self).__init__(root, split='unlabeled', download=download,
                                    transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        # if self.labels is not None:
        #     img, target = self.data[index], int(self.labels[index])
        # else:
        img, target = self.data[index], 0

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CelebA(data.Dataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM",
         "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U",
         "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
         "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0",
         "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c",
         "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk",
         "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root,
                 split="train",
                 target_type="attr",
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, **kwargs):
        import pandas
        super(CelebA, self).__init__()
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.transform = transform
        self.target_transform = target_transform

        if split.lower() == "train":
            split = 0
        elif split.lower() == "valid":
            split = 1
        elif split.lower() == "test":
            split = 2
        else:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="valid" or split="test"')

        with open(os.path.join(self.root, self.base_folder, "list_eval_partition.txt"), "r") as f:
            splits = pandas.read_csv(
                f, delim_whitespace=True, header=None, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "identity_CelebA.txt"), "r") as f:
            self.identity = pandas.read_csv(
                f, delim_whitespace=True, header=None, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "list_bbox_celeba.txt"), "r") as f:
            self.bbox = pandas.read_csv(
                f, delim_whitespace=True, header=1, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "list_landmarks_align_celeba.txt"), "r") as f:
            self.landmarks_align = pandas.read_csv(
                f, delim_whitespace=True, header=1)

        with open(os.path.join(self.root, self.base_folder, "list_attr_celeba.txt"), "r") as f:
            self.attr = pandas.read_csv(f, delim_whitespace=True, header=1)

        mask = (splits[1] == split)
        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(self.identity[mask].values)
        self.bbox = torch.as_tensor(self.bbox[mask].values)
        self.landmarks_align = torch.as_tensor(
            self.landmarks_align[mask].values)
        self.attr = torch.as_tensor(self.attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(
                self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(
            self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t))
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, 0  # target

    def __len__(self):
        return len(self.attr)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


# class CINIC10(ImageFolder):

    #   norm_mean = [0.5, 0.5, 0.5]
    # norm_std = [0.5, 0.5, 0.5]
    # if train:
    #     path = os.path.join(root, 'cinic10', 'train')
    # else:
    #     path = os.path.join(root, 'cinic10', 'valid')

    # print(path)
    # transform = [] if not train else [transforms.RandomCrop(32, padding=4, padding_mode='reflect')]
    # transform = transform + [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    # transform = transforms.Compose(transform)

    # dataset = ImageFolder(path, load_in_mem_labels=True, transform=transform)
