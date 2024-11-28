
import torch

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from util.utils import initialize_config

class RegularModule(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.denoising_model: torch.nn.Module = initialize_config(config["denoising_model"])
        self.classification_model: torch.nn.Module = initialize_config(config["classification_model"])

        self.signal_loss_fn = initialize_config(config["signal_loss_fn"])
        self.bit_loss_fn = torch.nn.BCELoss()

        self.use_wandb = self.config.get("use_wandb", False)

    def _log_statistics(self, loss, log_statistics, step_type):
        """
        Log the statistics of the current training or validation step.

        Args:
            loss: The computed loss value to be logged.
            log_statistics: A dictionary of other statistics to be logged.
            step_type: The step type, either 'train' or 'val' for logging purposes.
        """
        self.log(
            f"{step_type}/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        for key, value in log_statistics.items():
            self.log(f"{step_type}/{key}", value, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        noisy, clean, bit, _ = batch

        pred = self.denoising_model(noisy)
        pred_bit_prob = self.classification_model(pred)
        signal_loss = self.signal_loss_fn(pred, clean)
        bit_loss = self.bit_loss_fn(pred_bit_prob, bit)

        log_statistics = {
            "signal_loss": signal_loss,
            "bit_loss": bit_loss
        }
        loss = signal_loss + bit_loss

        self._log_statistics(loss, log_statistics, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean, bit, _ = batch

        pred = self.denoising_model(noisy)
        pred_bit_prob = self.classification_model(pred)
        signal_loss = self.signal_loss_fn(pred, clean)
        bit_loss = self.bit_loss_fn(pred_bit_prob, bit)

        loss = signal_loss + bit_loss

        pred_bit = (pred_bit_prob >= 0.5).int()
        BER = (pred_bit - bit).abs().mean()

        pred_bit_prob_with_clean = self.classification_model(clean)
        pred_bit_with_clean = (pred_bit_prob_with_clean >= 0.5).int()
        clean_BER = (pred_bit_with_clean - bit).abs().mean()

        log_statistics = {
            "signal_loss": signal_loss,
            "bit_loss": bit_loss,
            "BER": BER,
            "clean_BER": clean_BER
        }

        self._log_statistics(loss, log_statistics, "val")

        return loss

    def inference(self, batch):
        noisy, clean, bit, _ = batch

        pred = self.denoising_model(noisy.to(device=self.device))
        pred_bit_prob = self.classification_model(pred)
        pred_bit = (pred_bit_prob >= 0.5).int().cpu()
        BER = (pred_bit - bit).abs().mean()

        pred_bit_prob_with_clean = self.classification_model(clean.to(device=self.device))
        pred_bit_with_clean = (pred_bit_prob_with_clean >= 0.5).int().cpu()
        clean_BER = (pred_bit_with_clean - bit).abs().mean()

        log_statistics = {
            "BER": BER,
            "clean_BER": clean_BER,
        }

        return log_statistics
    

    def configure_optimizers(self):
        model_params = list(self.denoising_model.parameters()) + list(self.classification_model.parameters())

        optimizer = torch.optim.Adam(
            params=model_params,
            lr=self.config["optimizer"]["lr"],
            weight_decay=self.config["optimizer"]["weight_decay"],
            betas=(
                self.config["optimizer"]["beta1"],
                self.config["optimizer"]["beta2"],
            ),
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config["lr_scheduler"]["T_max"],
            eta_min= self.config["lr_scheduler"].get("eta_min", 0) 
        )
        return{
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        }

    def get_train_dataloader(self, trainset):
        return DataLoader(
            dataset=trainset,
            batch_size=self.config["train_dataloader"]["batch_size"],
            num_workers=self.config["train_dataloader"]["num_workers"],
            shuffle=self.config["train_dataloader"]["shuffle"],
            collate_fn=trainset.collate_fn,
            pin_memory=self.config["train_dataloader"]["pin_memory"],
        )

    def get_val_dataloader(self, valset):
        return DataLoader(
            dataset=valset,
            num_workers=self.config["val_dataloader"]["num_workers"],
            batch_size=self.config["val_dataloader"]["batch_size"],
            collate_fn=valset.collate_fn,
        )

    def get_test_dataloader(self, testset):
        return DataLoader(
            dataset=testset,
            num_workers=self.config["val_dataloader"]["num_workers"],
            batch_size=self.config["val_dataloader"]["batch_size"],
            collate_fn=testset.collate_fn,
        )
