import warnings
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")

from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from cdvae.common.utils import log_hyperparameters, PROJECT_ROOT


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Debug run with MP-20 dataset using specified Hydra jobs directory
    """
    # Force debug settings
    cfg.train.deterministic = True
    cfg.train.random_seed = 42
    
    # MP-20 specific data settings
    cfg.data.datamodule._target_ = "cdvae.data.datamodule.CrystDataModule"
    cfg.data.datamodule.name = "mp_20"
    cfg.data.datamodule.batch_size = 4
    cfg.data.datamodule.num_workers.train = 0
    cfg.data.datamodule.num_workers.val = 0
    cfg.data.datamodule.num_workers.test = 0
    cfg.data.prop = "formation_energy_per_atom"
    cfg.data.root = "/home/jamshid/workspace/nima/cdvae/hydra/singlerun/2025-04-08/pace_mp_20_v1/data/mp_20/"
    
    # Debug settings for trainer
    cfg.train.pl_trainer.fast_dev_run = False  # Run full training
    cfg.train.pl_trainer.gpus = 1  # Use GPU
    cfg.train.pl_trainer.max_epochs = 2
    cfg.train.pl_trainer.limit_train_batches = 2
    cfg.train.pl_trainer.limit_val_batches = 2
    cfg.train.pl_trainer.limit_test_batches = 2
    
    # Model specific debug settings for MP-20
    cfg.model._target_ = "cdvae.pl_modules.model.CDVAE"
    cfg.model.hidden_dim = 64
    cfg.model.latent_dim = 32
    cfg.model.max_atoms = 20  # MP-20 specific
    cfg.model.num_noise_level = 10
    cfg.model.encoder._target_ = "cdvae.pl_modules.gnn.DimeNetPlusPlusWrap"
    cfg.model.decoder._target_ = "cdvae.pl_modules.decoder.GemNetTDecoder"
    
    # Optimization debug settings
    cfg.optim.optimizer.lr = 0.001
    cfg.optim.use_lr_scheduler = False
    
    # Turn off wandb logging for debugging
    cfg.logging.wandb.mode = "disabled"
    
    # Set Hydra output directory
    hydra_dir = Path("/home/jamshid/workspace/nima/cdvae/hydra/singlerun/2025-04-08/pace_mp_20_v1")
    
    # Print debug configuration
    hydra.utils.log.info("\n=== Debug Configuration for MP-20 ===")
    hydra.utils.log.info(f"Dataset: MP-20")
    hydra.utils.log.info(f"Data path: {cfg.data.root}")
    hydra.utils.log.info(f"Hydra dir: {hydra_dir}")
    hydra.utils.log.info(f"Batch size: {cfg.data.datamodule.batch_size}")
    hydra.utils.log.info(f"Max epochs: {cfg.train.pl_trainer.max_epochs}")
    hydra.utils.log.info("=======================\n")

    # Rest of the original run function
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None
          
    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
