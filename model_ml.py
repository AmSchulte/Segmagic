import pytorch_lightning as lit
import segmentation_models_pytorch as smp
from TorchMetricLogger import TorchMetricLogger as TML
from TorchMetricLogger import TmlMean, TmlDice, TmlF1
import wandb
from model.losses import get_loss_function
import torch
from tqdm import tqdm
import pandas as pd


class Model(lit.LightningModule):
    def __init__(
            self, 
            settings
    ):
        super().__init__()
        self.model = self.get_architecture(settings["model"]["architecture"])(**settings["model"]["architecture_params"])
        
        self.steps_per_epoch = settings["spe"]
        self.num_epochs = settings["training"]["epochs"]
        self.best_valid_f1 = 0
        self.epoch_count = 0
        self.labels = settings["labels"]
        self.model_path = settings["model_path"]
        self.settings = settings

        # Initialize loss function using the factory
        self.loss = get_loss_function(
            settings["training"]["loss_name"],
            settings["training"]["loss_params"] if "loss_params" in settings["training"] else None
        )

        self.lr = settings["training"]["lr"]
        if settings["wandb_log"]["enabled"]:
            wandb.init(
                project=settings["wandb_log"]["project"], 
                entity=settings["wandb_log"]["entity"] if settings["wandb_log"]["entity"] else None
            )
            wandb.config.update(settings)
            self.tml = TML(log_function=wandb.log)
        else:
            self.tml = TML()
        
    def forward(self, x):
        return self.model(x)
    
    def step(self, batch):
        x, y, w = batch
        
        y_hat = self(x).contiguous()
        loss = self.loss(y_hat, y)
        p = y_hat.view([y.shape[0], 4, -1]).sigmoid().cpu().detach().numpy()
        l = y.view([y.shape[0], 4, -1]).cpu().detach().numpy()
        return loss, p, l

    def training_step(self, batch, batch_idx):
        loss, p, l = self.step(batch)

        self.tml(
            train_loss = TmlMean(values=loss),
            train_f1 = TmlF1(
                gold_labels=l, 
                predictions=p,
                class_names=self.labels
            )
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, p, l = self.step(batch)

        self.tml(
            val_loss = TmlMean(values=loss),
            val_f1 = TmlF1(
                gold_labels=l, 
                predictions=p,
                class_names=self.labels
            )
        )

        return loss
    
    def configure_optimizers(self):
        from model.optimizers import get_optimizer

        # With custom parameters
        opt = get_optimizer(
            self.settings["training"]["optimizer_name"],
            self.parameters(), 
            lr=self.settings["training"]["lr"],
            optimizer_params=self.settings["training"]["optimizer_params"] if "optimizer_params" in self.settings["training"] else None
        )
        
        # FIXME: put this into the toml file
        if False:
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.num_epochs4,
            )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.steps_per_epoch * self.num_epochs,
            eta_min=self.lr / 1e5
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }
    
    def get_architecture(self, name):
        """
        Get the architecture of the model.
        
        Returns:
            The architecture of the model as a string
        """
        if name.lower() == "segformer":
            from segmentation_models_pytorch import Segformer
            return Segformer
        elif name.lower() == "unet":
            from segmentation_models_pytorch import Unet
            return Unet
        elif name.lower() == "scunet":
            from model.single_channel_model import SCUnet
            return SCUnet
        elif name.lower() == "manet":
            from segmentation_models_pytorch import MAnet
            return MAnet
        else:
            raise ValueError(f"Unknown architecture: {name}. Supported architectures are: 'segformer', 'unet', 'scunet'.")

    def on_train_epoch_end(self):
        result = self.tml.on_batch_end()
        
        self.epoch_count += 1
        current_valid_f1 = result['val_f1_soft_micro']
        if current_valid_f1 > self.best_valid_f1:
            self.best_valid_f1 = current_valid_f1
            tqdm.write(f"\nSaving model for epoch {self.epoch_count} for best soft validation F1: {self.best_valid_f1}")
            #torch.save(self.model.state_dict(), 'output/best_model.pth')
            torch.save(self.model, self.model_path)
            result_df = pd.DataFrame.from_dict(result, orient='index')
            result_df.to_excel(self.model_path[:-3]+'xlsx')