import numpy as np
import torch as T
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math 
from itertools import chain
import argparse
from torch.nn.modules.activation import ReLU
import wandb
import sys
import torchvision
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from data import MNISTDataModule
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from datetime import datetime
import shutil


class SemanticBusArchitecture(pl.LightningModule):
    def __init__(self, n_modules, pointer_dim, content_dim, n_cycles=1, log_channels=[]):
        super().__init__()

        # CONSTANTS & FLAGS
        self.automatic_optimization = False
        self.print_channels = log_channels
        self.n_modules = n_modules
        self.n_cycles = n_cycles
        self.content_dim = content_dim
        self.pointer_dim = pointer_dim
        self.color_channels = 1

        # EYES
        """self.cnn_encoder = nn.Sequential(
            nn.Conv2d(self.color_channels, 64, 3, 1, 0),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 16, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3, 1, 0),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, pointer_dim+content_dim, 1, 1, 0),
            nn.ReLU()
        )
        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(content_dim, 16, 1, 1, 0),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=3),
            nn.Conv2d(64, 16, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 0),
        )"""

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(self.color_channels, 16, 3, 1, 0),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 0),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, pointer_dim+content_dim, 1, 1, 0),
            nn.ReLU(True)
        )

        self.cnn_decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(content_dim, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 1, 3, 1, 0),
            nn.UpsamplingBilinear2d(size=(28,28))
        )

        # INIT PARAMETERS AND OPTIMIZERS
        self.configure_optimizers()

    def configure_optimizers(self):
        eye_opti = T.optim.Adam(chain(
            *[module.parameters() for module in [self.cnn_encoder, self.cnn_decoder]]))
        return eye_opti

    def print(self, *args, channel="debug"):
        if channel in self.print_channels:
            print(*args)

    def training_step(self, batch, batch_idx):
        inputs, _ = batch

        # PERCEIVE
        encoder_embeds = self.cnn_encoder(inputs)

        self.print("encoder_embeds.shape", encoder_embeds.shape, channel="shapes")
        _, in_contents = encoder_embeds.flatten(-2).transpose(-2,-1).split([self.pointer_dim, self.content_dim], dim=-1)

        # RECONSTRUCT
        reshaped_returned_contents = in_contents.transpose(-2, -1).reshape(
            inputs.shape[0], self.content_dim, *encoder_embeds.shape[-2:])
        self.print(f"reshaped_returned_contents.shape {reshaped_returned_contents.shape}", channel="shapes")
        self.print(f"reshaped_returned_contents==encoder_embeds[:,self.pointer_dim:] \
            {reshaped_returned_contents==encoder_embeds[:,self.pointer_dim:]}", channel="shapes")
        #imagination = self.cnn_decoder(reshaped_returned_contents)
        
        contents = encoder_embeds[:,self.pointer_dim:,:,:]
        self.print(f"contents.shape {contents.shape}", channel="shapes")
        imagination = T.sigmoid(self.cnn_decoder(contents))
        #print(imagination.max())
        # ZERO GRAD
        opt = self.optimizers()
        opt.zero_grad()

        # LOSS
        #recon_loss = F.mse_loss(effective_returned_contents, in_contents)
        recon_loss = F.mse_loss(imagination, inputs)

        # OPTIMIZE
        self.manual_backward(recon_loss)
        opt.step()

        # LOGGING
        #self.print(f"GRAD self.receivers.grad.sum() {self.receivers.grad.sum()}", channel="grad-main")
        self.print(f"LOSS {recon_loss}", channel="loss")
        self.print(f"INPUT VALUE RANGE {inputs.min(), inputs.max()}", channel="values")
        self.log("recon_loss", recon_loss)

        #VIZ
        if not batch_idx%100:
            self.viz(batch_idx, inputs, imagination)


    def viz(self, name, *batch_arrays):
        rows_concat = np.concatenate([batch.detach().cpu().numpy() for batch in batch_arrays], axis=-2)
        column_concat = np.concatenate(rows_concat, axis=-1)
        result = column_concat.transpose(1,2,0)
        plt.imsave(args.resultpath+f"{name}.png", result.squeeze())


    def get_dummy_set(self, size=1000, n_tokens=16):
        pointers = T.randn(size, n_tokens, self.pointer_dim)
        contents = T.randn(size, n_tokens, self.content_dim)
        dummy_set = pointers, contents
        return dummy_set

        
if __name__=="__main__":
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    argstr = " ".join(sys.argv)
    resultpath = f"results/_tmp/"
    os.makedirs(resultpath, exist_ok=True)
    with open(resultpath+"log.txt", "w") as file:
        file.write(argstr)

    # INIT ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true")
    parser.add_argument("--archive", type=str, default="")
    parser.add_argument("--logch", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--pointer-dims", type=int, default=16)
    parser.add_argument("--content-dims", type=int, default=32)
    parser.add_argument("--modules", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--nThreads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    args.logch = args.logch.split(",")
    args.resultpath = resultpath
    print("ARGS", args.__dict__)
    T.manual_seed(args.seed)

    # INIT LOGGER
    wandblogger = WandbLogger(project='sba', entity='aharter', notes=args.notes, config=args.__dict__, save_dir="saves")

    # INIT DATA
    if args.dataset=="mnist":
       dm = MNISTDataModule("../data")

    # INIT MODEL
    sba = SemanticBusArchitecture(
        args.modules, 
        args.pointer_dims, 
        args.content_dims, 
        log_channels=args.logch, 
        n_cycles=args.cycles)

    # DUMMY DEBUG
    if args.overfit:
        trainer = pl.Trainer(logger=wandblogger, overfit_batches=args.overfit)
        trainer.fit(sba, dm)

    # TRAINING
    if args.train:
        trainer = pl.Trainer(logger=wandblogger)
        trainer.fit(sba, dm)

    # ARCHIVE
    if args.archive:
        name = timestamp if args.archive=="time" else args.archive
        archivepath = f"results/{name}/"
        os.makedirs(archivepath)
        for file in os.listdir(resultpath):
            shutil.move(resultpath+file, archivepath+file)
            shutil.copy(os.curdir+"/"+__file__, archivepath+__file__)
    
