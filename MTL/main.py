from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from pytorch_lightning import loggers as pl_loggers
from Loss_function import Class_wise_Dice_score_multi_task_learning, Class_Wise_TIL_Detection_FROC_multi_task_learning
print(torch.version.cuda)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from DeepLabv3_plus import DeepLabv3Plus
from sklearn.model_selection import KFold
tb_loggers = pl_loggers.TensorBoardLogger("logs/")


def one_hot_label_seg(
        labels: torch.Tensor,
        num_classes: int = 3,
        device=torch.device('cuda:0'),
        dtype=torch.int32,
        eps: float = 1e-6,
        ignore_index=250,
) -> torch.Tensor:
    shape = labels.shape
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret

def one_hot_label_det(
        labels: torch.Tensor,
        num_classes: int = 2,
        device=torch.device('cuda:0'),
        dtype=torch.int32,
        eps: float = 1e-6,
        ignore_index=250,
) -> torch.Tensor:
    shape = labels.shape
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret


class SemSegment(LightningModule):
    def __init__(
            self,
            lr: float = 0.01,
            num_classes: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            after_training: bool = True,
            **kwargs
    ):
        """Basic model for semantic segmentation. Uses UNet architecture by default.

        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

        Implemented by:

            - `Annika Brundyn <https://github.com/annikabrundyn>`_

        Args:
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr
        self.after_training = True
        self.net = DeepLabv3Plus()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask, mask_name = batch
        img = img.float()
        batch_size = len(mask_name)
        mask = mask.long()

        label_seg = one_hot_label_seg(mask)
        label_det = one_hot_label_det(mask)

        out_seg, out_det = self(img)

        dice0, dice1, dice2, diceT, bce_loss, dice_loss, loss_val_seg = Class_wise_Dice_score_multi_task_learning(out_seg, label_seg, mask_name, batch_size)
        loss_val_det, froc_score = Class_Wise_TIL_Detection_FROC_multi_task_learning(out_det, label_det, mask_name, batch_size)

        loss_val = (0.13 * loss_val_seg) + (0.87 * loss_val_det)

        # loss_val = F.cross_entropy(out, mask, ignore_index=250)
        # self.log('train_dice1', val_dice, on_step=False, on_epoch=True)
        #self.log('train_dice2', dice, on_step=False, on_epoch=True)
        self.log('train_dice3', diceT, on_step=False, on_epoch=True)
        self.log('train_loss', loss_val, on_step=False, on_epoch=True)
        self.log('train_bce_loss', bce_loss, on_step=False, on_epoch=True)
        self.log('train_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('train_dice_other', dice0, on_step=False, on_epoch=True)
        self.log('train_dice_tumor', dice1, on_step=False, on_epoch=True)
        self.log('train_dice_stroma', dice2, on_step=False, on_epoch=True)
        self.log('train_loss_seg', loss_val_seg, on_step=False, on_epoch=True)
        self.log('train_loss_det', loss_val_det, on_step=False, on_epoch=True)
        self.log('train_froc', float(froc_score), on_step=False, on_epoch=True)

        # log_dict = {"train_loss": loss_val}
        # return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}



        return loss_val

    def validation_step(self, batch, batch_idx):
        img, mask, mask_name = batch
        img = img.float()
        mask = mask.long()
        batch_size = len(mask_name)
        label_seg = one_hot_label_seg(mask)
        label_det = one_hot_label_det(mask)
        out_seg, out_det = self(img)
        pred_seg = F.softmax(out_seg)
        pred_det = F.softmax(out_det)

        dice0, dice1, dice2, diceT, bce_loss, dice_loss, loss_val_seg = Class_wise_Dice_score_multi_task_learning(out_seg, label_seg, mask_name, batch_size)
        loss_val_det, froc_score = Class_Wise_TIL_Detection_FROC_multi_task_learning(out_det, label_det, mask_name, batch_size)

        loss_val = (0.13 * loss_val_seg) + (0.87 * loss_val_det)

        # loss_val = F.cross_entropy(out, mask, ignore_index=250)
        # self.log('val_dice1', val_dice, on_step=False, on_epoch=True, sync_dist=True)
        #self.log('val_dice2', dice, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_dice3', diceT, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_bce_loss', bce_loss, on_step=False, on_epoch=True)
        self.log('val_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('val_dice_other', dice0, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_dice_tumor', dice1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_dice_stroma', dice2, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss_seg', loss_val_seg, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss_det', loss_val_det, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_froc', float(froc_score), on_step=False, on_epoch=True, sync_dist=True)

        # print(loss_val)
        # return {"val_loss": loss_val}

        if (batch_idx == 0) and self.after_training:
            self.logger.experiment.add_image("val_input", make_grid(img, nrow=5))
            self.logger.experiment.add_image("val_label_seg", make_grid(label_seg[:, :3], nrow=5))
            self.logger.experiment.add_image("val_label_det", make_grid(label_det[:, :3], nrow=5))
            self.logger.experiment.add_image("val_pred_seg", make_grid(pred_seg[:, :3], nrow=5))
            self.logger.experiment.add_image("val_pred_det", make_grid(pred_det[:, :3], nrow=5))
        return loss_val

    def training_epoch_end(self, outputs):
        self.after_training = True

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay = 0.001)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument(
            "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
        )

        return parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cli_main():
    from Datamodule import DataModule

    seed_everything(1234)
    #kfold = KFold(n_splits=5, shuffle=True)

    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = SemSegment.add_model_specific_args(parser)
    # datamodule args
    parser = KittiDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    args.__dict__["gpus"] = 1
    args.__dict__["batch_size"] = 5
    args.__dict__["precision"] = 32
    args.__dict__["logger"] = tb_loggers
    args.__dict__["max_epochs"] = 300
    args.__dict__["num_workers"] = 20
    args.__dict__["callbacks"] = [ModelCheckpoint(save_top_k=1, save_last=False, monitor="val_loss"),
                                  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                  LearningRateMonitor()]  # , EarlyStopping(monitor="val_loss", mode="min", patience=5 )]


    # data
    dm = DataModule(args.data_dir).from_argparse_args(args)

    # model
    model = SemSegment(**args.__dict__)
    print("total parameter count:", count_parameters(model))
    # train
    trainer = Trainer().from_argparse_args(args)
    trainer.fit(model, dm)
    model.eval()
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()


