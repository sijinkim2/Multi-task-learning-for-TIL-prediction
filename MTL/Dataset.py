import os

import numpy as np
from torch.utils.data import Dataset

from pl_bolts.utils import _PIL_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")

#segmentation
#DEFAULT_VALID_LABELS = (0, 1, 2)

#detection
DEFAULT_VALID_LABELS = (0, 1)

class train_Dataset(Dataset):
    """
    The `encode_segmap` function sets all pixels with any of the `void_labels` to `ignore_index`
    (250 by default). It also sets all of the valid pixels to the appropriate value between 0 and
    `len(valid_labels)` (since that is the number of valid classes), so it can be used properly by
    the loss function when comparing with the output.
    """

    IMAGE_PATH_seg = os.path.join("/path/to/segmentation/train/dataset/img")
    MASK_PATH_seg = os.path.join("/path/to/segmentation/train/dataset/mask")
    IMAGE_PATH_det = os.path.join("/path/to/detection/train/dataset/img")
    MASK_PATH_det = os.path.join("/path/to/detection/train/dataset/mask")
    def __init__(
        self,
        data_dir: str,
        img_size: tuple = (256, 256),
        #void_labels: list = DEFAULT_VOID_LABELS,
        valid_labels: list = DEFAULT_VALID_LABELS,
        transform=None,
    ):
        """
        Args:
            data_dir (str): where to load the data from path, i.e., '/path/to/folder/with/data_semantics/'
            img_size: image dimensions (width, height)
            void_labels: useless classes to be excluded from training
            valid_labels: useful classes to include
        """
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `PIL` which is not installed yet.")

        self.img_size = img_size
        #self.void_labels = void_labels
        self.valid_labels = valid_labels
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform

        self.data_dir = data_dir
        self.img_path_seg = os.path.join(self.data_dir, self.IMAGE_PATH_seg)
        self.mask_path_seg = os.path.join(self.data_dir, self.MASK_PATH_seg)
        self.img_path_det = os.path.join(self.data_dir, self.IMAGE_PATH_det)
        self.mask_path_det = os.path.join(self.data_dir, self.MASK_PATH_det)
        self.seg_img_list = self.get_filenames(self.img_path_seg)
        self.seg_mask_list = self.get_filenames(self.mask_path_seg)
        self.det_img_list = self.get_filenames(self.img_path_det)
        self.det_mask_list = self.get_filenames(self.mask_path_det)
        self.img_list = self.seg_img_list + self.det_img_list
        self.mask_list = self.seg_mask_list + self.det_mask_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        mask = Image.open(self.mask_list[idx]).convert("L")
        mask_name = os.path.join(self.mask_list[idx][19])
        ##print(mask_name)
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        mask = self.encode_segmap(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask, mask_name


    def get_filenames(self, path):
        """ Returns a list of absolute paths to images inside a given path."""
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


    def encode_segmap(self, mask):
        """ Sets void classes to zero so they won't be considered for training."""
        #for voidc in self.void_labels:
        #    mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        # remove extra idxs from updated dataset
        #mask[mask > 18] = self.ignore_index
        return mask



class val_Dataset(Dataset):

    IMAGE_PATH_seg = os.path.join("/path/to/segmentation/val/dataset/img")
    MASK_PATH_seg = os.path.join("/path/to/segmentation/val/dataset/mask")
    IMAGE_PATH_det = os.path.join("/path/to/detection/train/dataset/img")
    MASK_PATH_det = os.path.join("/path/to/detection/train/dataset/mask")

    def __init__(
            self,
            data_dir: str,
            img_size: tuple = (256, 256),
            # void_labels: list = DEFAULT_VOID_LABELS,
            valid_labels: list = DEFAULT_VALID_LABELS,
            transform=None,
    ):
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `PIL` which is not installed yet.")

        self.img_size = img_size
        # self.void_labels = void_labels
        self.valid_labels = valid_labels
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform

        self.data_dir = data_dir
        self.img_path_seg = os.path.join(self.data_dir, self.IMAGE_PATH_seg)
        self.mask_path_seg = os.path.join(self.data_dir, self.MASK_PATH_seg)
        self.img_path_det = os.path.join(self.data_dir, self.IMAGE_PATH_det)
        self.mask_path_det = os.path.join(self.data_dir, self.MASK_PATH_det)
        self.seg_img_list = self.get_filenames(self.img_path_seg)
        self.seg_mask_list = self.get_filenames(self.mask_path_seg)
        self.det_img_list = self.get_filenames(self.img_path_det)
        self.det_mask_list = self.get_filenames(self.mask_path_det)
        self.img_list = self.seg_img_list + self.det_img_list
        self.mask_list = self.seg_mask_list + self.det_mask_list


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        mask = Image.open(self.mask_list[idx]).convert("L")
        mask_name = os.path.join(self.mask_list[idx][19])
        #print(mask_name)
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        mask = self.encode_segmap(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask, mask_name

    def get_filenames(self, path):
        """ Returns a list of absolute paths to images inside a given path."""
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list

    def encode_segmap(self, mask):
        """ Sets void classes to zero so they won't be considered for training."""
        # for voidc in self.void_labels:
        #    mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        # remove extra idxs from updated dataset
        # mask[mask > 18] = self.ignore_index
        return mask

# test dataset
class Total_Dataset(Dataset):
        IMAGE_PATH = os.path.join("/path/to/test/images")
        MASK_PATH = os.path.join("/path/to/test/masks")

        def __init__(
                self,
                data_dir: str,
                img_size: tuple = (256, 256),
                # void_labels: list = DEFAULT_VOID_LABELS,
                valid_labels: list = DEFAULT_VALID_LABELS,
                transform=None,
        ):
            if not _PIL_AVAILABLE:  # pragma: no cover
                raise ModuleNotFoundError("You want to use `PIL` which is not installed yet.")

            self.img_size = img_size
            # self.void_labels = void_labels
            self.valid_labels = valid_labels
            self.ignore_index = 250
            self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
            self.transform = transform

            self.data_dir = data_dir
            self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)  # train img path
            self.mask_path = os.path.join(self.data_dir, self.MASK_PATH)  # train mask path
            self.img_list = self.get_filenames(self.img_path)
            self.mask_list = self.get_filenames(self.mask_path)

        def __len__(self):
            return len(self.img_list)

        def __getitem__(self, idx):
            img = Image.open(self.img_list[idx])
            img = img.resize(self.img_size)
            img = np.array(img)

            mask = Image.open(self.mask_list[idx]).convert("L")
            mask = mask.resize(self.img_size)
            mask = np.array(mask)
            mask = self.encode_segmap(mask)

            if self.transform:
                img = self.transform(img)

            return img, mask

        def get_filenames(self, path):
            """ Returns a list of absolute paths to images inside a given path."""
            files_list = list()
            for filename in os.listdir(path):
                files_list.append(os.path.join(path, filename))
            return files_list

        def encode_segmap(self, mask):
            """ Sets void classes to zero so they won't be considered for training."""
            # for voidc in self.void_labels:
            #    mask[mask == voidc] = self.ignore_index
            for validc in self.valid_labels:
                mask[mask == validc] = self.class_map[validc]
            # remove extra idxs from updated dataset
            # mask[mask > 18] = self.ignore_index
            return mask
