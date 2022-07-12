import argparse
from pprint import pprint
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import Dataset, DataLoader
from sys import platform

class PalsarDataset(Dataset):
    def __init__(self, dataset_dir, nchannels):
        self.dataset = np.load(dataset_dir).astype(np.single)
        self.label = self.dataset[:,:,:,nchannels]>0
        self.label = self.label[:,np.newaxis,:,:]
        self.images = self.dataset[:, :, :, :nchannels].transpose((0,3,1,2))

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        label = self.label[idx,:,:,:]
        return image, label

def get_dateset(batch_size, data, nchannels):
    if platform == "darwin":
        base_path = 'dataset/'
    else:
        base_path = '/geoinfo_vol1/zhao2/proj2_dataset/'

    if data == 'palsar':
        train_dataset = PalsarDataset(base_path+'proj2_train_' + str(nchannels) + 'chan.npy', nchannels)
        val_dataset = PalsarDataset(base_path+'proj2_val_' + str(nchannels) + 'chan.npy', nchannels)
    else:
        train_dataset = PalsarDataset(base_path+'proj2_train_' + str(nchannels) + 'chan_s1.npy', nchannels)
        val_dataset = PalsarDataset(base_path+'proj2_val_' + str(nchannels) + 'chan_s1.npy', nchannels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    return train_dataloader, val_dataloader


class SegModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, learning_rate, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, encoder_weights="imagenet", in_channels=in_channels, classes=out_classes, **kwargs
        )
        self.model = self.model.float()
        self.best_loss = float('inf')

        params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.learning_rate = learning_rate
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, mask, mode="binary", threshold=0.5)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean().item()

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        per_image_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_per_image_f1": per_image_f1,
            f"{stage}_dataset_f1": dataset_f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-p', type=str, help='Load trained weights')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-bb', type=str, help='backbone')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-data', type=str, help='dataset used')
    parser.add_argument('-nc', type=int, help='num of channels')
    args = parser.parse_args()
    model_name = args.m
    load_weights = args.p
    backbone = args.bb
    batch_size=args.b
    MAX_EPOCHS=300
    fine_tune=False
    learning_rate = args.lr
    data = args.data
    nchannels = args.nc
    weight_decay = learning_rate/10

    train_dataloader, val_dataloader = get_dateset(batch_size, data, nchannels)

    wandb.login(key='203b00f27d58654c3c411c11374267c050d68120')
    wandb_logger = WandbLogger(project='proj2_palsar_torch', log_model='all', name='model_name' + str(model_name) + 'backbone_'+ str(backbone)+ 'batchsize_'+str(batch_size)+'learning_rate_'+str(learning_rate)+'_data_'+str(data)+'_nchannels_'+str(nchannels))
    model = SegModel(arch=model_name, encoder_name=backbone, in_channels=nchannels, out_classes=1, learning_rate=learning_rate)


    if platform == "darwin":
        save_base_path = 'proj2_model_torch/'
    else:
        save_base_path = '/geoinfo_vol1/zhao2/proj2_model_torch/'

    save_path = save_base_path + 'proj2_' + model_name + '_pretrained_' + backbone + '_dataset_' + data + '_nchannels_' + str(nchannels)
    if load_weights=='no':
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_top_k=1, monitor="val_loss")
        trainer = pl.Trainer(
            gpus=0,
            max_epochs=MAX_EPOCHS,
            logger=wandb_logger
        )
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        valid_metrics = trainer.validate(model, dataloaders=val_dataloader, verbose=False)
        pprint(valid_metrics)
    else:
        model = SegModel.load_from_checkpoint(save_path, arch=model_name, encoder_name=backbone, in_channels=nchannels, out_classes=1, learning_rate=learning_rate)
        print(model.learning_rate)