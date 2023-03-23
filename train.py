import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from datamodule import FashionMNISTDataModule
from model import FashionMNISTClassifier

@hydra.main(config_name="config")
def train(cfg: DictConfig):
    data_module = FashionMNISTDataModule(batch_size=cfg.batch_size)
    model = FashionMNISTClassifier(lr=cfg.lr, optimizer_name=cfg.optimizer)
    trainer = pl.Trainer(max_epochs=cfg.epochs)
    trainer.fit(model, data_module) 


train()

