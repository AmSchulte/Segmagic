import pytorch_lightning as lit
import segmentation_models_pytorch as smp
from TorchMetricLogger import TorchMetricLogger as TML
from TorchMetricLogger import TmlMean, TmlDice, TmlF1
import wandb
from model.losses import get_loss_function
import torch
from tqdm import tqdm
import pandas as pd
from torch.nn import functional as F

class Model(lit.LightningModule):
    def __init__(
            self, 
            settings
    ):
        super().__init__()
        
        # Set deep supervision early so it's available for get_architecture
        self.deep_supervision = settings["model"]["architecture_params"].get("deep_supervision", False)
        self.deep_supervision_weight = settings["model"]["architecture_params"].get("deep_supervision_weight", 0.5)
        
        self.model = self.get_architecture(settings["model"]["architecture"])(**settings["model"]["architecture_params"])
        
        # replace the segmentation head with a custom one
        from model.blocks.heads import SegmentationHead, DeepSupervisionHead
        self.model.segmentation_head = SegmentationHead(
            settings["model"]["architecture_params"]["decoder_channels"][-1], 
            settings["model"]["architecture_params"]["classes"],
            activation=settings["model"]["architecture_params"]["activation"],
        )
        
        if self.deep_supervision:
            # Create auxiliary heads for intermediate decoder outputs
            decoder_channels = settings["model"]["architecture_params"]["decoder_channels"]
            num_classes = settings["model"]["architecture_params"]["classes"]
            
            self.aux_heads = torch.nn.ModuleList([
                DeepSupervisionHead(
                    in_channels=decoder_channels[i], 
                    out_channels=num_classes
                ) 
                for i in range(len(decoder_channels) - 1)  # Exclude the final decoder layer
            ])
        
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
        # Encode and decode
        feats = self.model.encoder(x)
        decs = self.model.decoder(feats)
        # Ensure list
        decs = decs if isinstance(decs, list) else [decs]
        
        # Apply segmentation head to all
        outs = [self.model.segmentation_head(decs[0])]
        
        if self.deep_supervision and self.training:
            outs = outs + [head(d) for head, d in zip(self.aux_heads, decs[1:])]
            return outs
        
        return outs[0]
    
    def step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        # Wrap single output into list
        preds = y_hat if isinstance(y_hat, list) else [y_hat]
        
        # Compute losses
        main_pred = preds[0].contiguous()
        main_loss = self.loss(main_pred, y)
        
        if self.deep_supervision and self.training and len(preds) > 1:
            aux_preds = preds[1:]
            aux_weight = self.deep_supervision_weight / len(aux_preds)
            aux_losses = []
            for p in aux_preds:
                # Downsample mask to prediction size
                target_size = p.shape[2:]
                y_ds = F.interpolate(y, size=target_size, mode='nearest')
                aux_losses.append(self.loss(p.contiguous(), y_ds))
            total_aux = sum(aux_losses)
            loss = (1 - self.deep_supervision_weight) * main_loss + aux_weight * total_aux
        else:
            loss = main_loss
            aux_losses = [0.0] * len(self.aux_heads)

        # Metrics from main output
        p = main_pred.sigmoid().view(y.shape[0], 4, -1).detach().cpu().numpy()
        l = y.view(y.shape[0], 4, -1).cpu().numpy()
        return loss, main_loss, aux_losses, p, l

    def training_step(self, batch, batch_idx):
        loss, main_loss, aux_losses, p, l = self.step(batch)
        current_lr = self.optimizers().param_groups[0]['lr']

        self.tml(
            train_loss = TmlMean(values=loss),
            train_f1 = TmlF1(
                gold_labels=l, 
                predictions=p,
                class_names=self.labels
            ),
            train_main_loss = TmlMean(values=main_loss),
            train_aux_losses_0 = TmlMean(values=aux_losses[0]),
            train_aux_losses_1 = TmlMean(values=aux_losses[1]),
            train_aux_losses_2 = TmlMean(values=aux_losses[2]),
            lr = TmlMean(values=current_lr)
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, main_loss, aux_losses, p, l = self.step(batch)

        self.tml(
            val_loss = TmlMean(values=loss),
            val_f1 = TmlF1(
                gold_labels=l, 
                predictions=p,
                class_names=self.labels
            ),
            main_loss = TmlMean(values=main_loss)
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
            eta_min=self.lr / 1e4
        )   
        # decay at 60 and 80 percent of the training
        #sched = torch.optim.lr_scheduler.MultiStepLR(
        #    opt,
        #    milestones=[int(self.steps_per_epoch * self.num_epochs * 0.6), int(self.steps_per_epoch * self.num_epochs * 0.8)],
        #    gamma=0.1
        #)

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
            #from segmentation_models_pytorch import Segformer
            # add 0.1 dropout to the Segformer
            from model.blocks.decoders.segformer.model import Segformer
            return Segformer
        elif name.lower() == "unet":
            from model.blocks.decoders.unet.model import Unet
            return Unet
        elif name.lower() == "scunet":
            from model.single_channel_model import SCUnet
            return SCUnet
        elif name.lower() == "manet":
            from segmentation_models_pytorch import MAnet
            return MAnet
        elif name.lower() == "unetplusplus":
            from segmentation_models_pytorch import UnetPlusPlus
            return UnetPlusPlus
        elif name.lower() == "linknet":
            from segmentation_models_pytorch import Linknet
            return Linknet
        elif name.lower() == "fpn":
            from segmentation_models_pytorch import FPN
            return FPN
        elif name.lower() == "pspnet":
            from segmentation_models_pytorch import PSPNet
            return PSPNet
        elif name.lower() == "deeplabv3plus":
            from segmentation_models_pytorch import DeepLabV3Plus
            return DeepLabV3Plus    
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