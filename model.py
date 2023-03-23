import torch
from torch import nn, optim   
import pytorch_lightning as pl 

class FashionMNISTClassifier(pl.LightningModule):
    def __init__(self, lr: float, optimizer_name: str):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.25)
        self.output = nn.Linear(64, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.lr = lr
        self.optimizer_name = optimizer_name

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.output(x)
        x = self.log_softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.nll_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.nll_loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.nll_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer")
        return optimizer
