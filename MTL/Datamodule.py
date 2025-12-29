# type: ignore[override]
import os
from typing import Any, Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from Dataset import train_Dataset, val_Dataset
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class DataModule(LightningDataModule):

    name = "kitti"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: float = 0.2,
        test_split: float = 0.0,
        num_workers: int = 8,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last


        #학습 실행과 동시에 데이터 스플릿
        '''
        # split into train, val, test
        train_dataset = KittiDataset(self.data_dir, transform=self._default_transforms())
        #test_dataset = test_Dataset(self.data_dir, transform=self._default_transforms())

        val_len = round(val_split * len(train_dataset))
        test_len = round(test_split * len(train_dataset))
        train_len = len(train_dataset) - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(
            train_dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(self.seed)
        )
        '''

        #이미 스플릿된 데이터셋
        self.trainset = train_Dataset(self.data_dir, transform=self._default_transforms())
        self.valset = val_Dataset(self.data_dir, transform=self._default_transforms())


        '''
        kfold = KFold(n_splits=5, shuffle= True)

        self.trainset = KittiDataset(self.data_dir, transform=self._default_transforms())
        self.valset = val_Dataset(self.data_dir, transform=self._default_transforms())
        self.dataset = ConcatDataset([self.trainset, self.valset])

        print("---------------------------------------------------------")

        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset)):
            print(f'FOLD {fold}')
            print("---------------------------------------------------------")

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)


    def train_dataloader(self) -> DataLoader:
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size, sampler=train_subsampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size, sampler=val_subsampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader
    '''

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    '''
    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader
    '''
    def _default_transforms(self) -> Callable:
        kitti_transforms = transforms.Compose(
            [
                #transforms.RandomApply([transforms.transforms.HEDJitter(theta=0.05)], p = 0.8),
                #transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                #transforms.RandomGrayscale(p=0.2),
                #transforms.RandomResizedCrop(512),

                #transforms.Normalize(
                #    mean=[0.35675976, 0.37380189, 0.3764753], std=[0.32064945, 0.32098866, 0.32325324]
                #),

            ]
        )
        return kitti_transforms
