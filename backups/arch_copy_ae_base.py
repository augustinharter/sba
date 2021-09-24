import shutil
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
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data import MNISTDataModule
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os, sys
from datetime import datetime


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
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 0),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, pointer_dim+content_dim, 1, 1, 0),
            nn.ReLU()
        )
        self.cnn_decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(content_dim, 16, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 1, 3, 1, 0),
            nn.UpsamplingBilinear2d(size=(28,28))
        )

        # MODULES CONTENTS
        self.receivers = nn.Parameter(T.zeros(n_modules, pointer_dim))
        self.memories = nn.Parameter(T.zeros(n_modules, content_dim, content_dim))
        self.routing_transforms = [nn.Sequential(
            nn.Linear(content_dim+pointer_dim, pointer_dim),
            nn.ReLU(),
            nn.Linear(pointer_dim, pointer_dim),
            nn.ReLU()
        ) for _ in range(n_modules)]
        self.bus_pointer = T.zeros(n_modules, pointer_dim)
        self.bus_contents = T.zeros(n_modules, content_dim)

        # INIT PARAMETERS AND OPTIMIZERS
        self.reset_parameters()
        self.configure_optimizers()


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.receivers, a=math.sqrt(5))
        init.kaiming_uniform_(self.memories, a=math.sqrt(5))
        self.print("memories init:", self.memories, channel="init")


    def configure_optimizers(self):
        routing_transforms_optimizer = T.optim.Adam(chain(
            *[module.parameters() for module in self.routing_transforms]))
        eye_opti = T.optim.Adam(chain(
            *[module.parameters() for module in [self.cnn_encoder, self.cnn_decoder]]))
        memories_optimizer = T.optim.Adam([self.memories])
        receivers_optimizer = T.optim.Adam([self.receivers])
        return routing_transforms_optimizer, memories_optimizer, receivers_optimizer, eye_opti


    def step_all_optimizers(self):
        self.routing_transforms_optimizer.step()
        self.memories_optimizer.step()
        self.receivers_optimizer.step()


    def zero_all_optimizers(self):
        self.routing_transforms_optimizer.zero_grad()
        self.memories_optimizer.zero_grad()
        self.receivers_optimizer.zero_grad()


    def print(self, *args, channel="debug"):
        if channel in self.print_channels:
            print(*args)


    def pass_through_modules(self, in_pointers:T.Tensor, in_contents:T.Tensor, batched_memories:T.Tensor):
        bsize, n_tok, pdim = in_pointers.shape
        n_mod = self.receivers.shape[0]
        cdim = in_contents.shape[-1]
        self.print(f"batchsize {bsize}", channel="shapes")

        # WEIGHTINGS
        self.print(f"in_pointers.shape {in_pointers.shape}", channel="shapes")
        self.print(f"self.receivers.shape {self.receivers.shape}", channel="shapes")
        #TODO try squashing function instead
        raw_weightings = F.cosine_similarity(self.receivers.view(    1, n_mod,     1, pdim), 
                                                in_pointers.view(bsize,     1, n_tok, pdim), 
                                            dim=-1)
        #raw_weightings = self.receivers.matmul(in_pointers.transpose(-2, -1))
        self.print(f"raw_weightings.shape {raw_weightings.shape}", channel="shapes")
        self.print("raw weights sum", raw_weightings.sum(), channel="grad")
        activated_weightings = (raw_weightings.abs())
        self.print("activated weights sum", activated_weightings.sum(), channel="grad")
        self.print(f"activated_weightings.shape {activated_weightings.shape}", channel="shapes")
        normed_weightings = activated_weightings/activated_weightings.sum(dim=1, keepdim=True)
        self.print("normed_weightings.shape", normed_weightings.shape, normed_weightings.sum(0).sum(-1), channel="shapes")

        # CONTENTS
        effective_in_contents = (normed_weightings.view(bsize, n_mod, n_tok,    1) 
                                     * in_contents.view(bsize,     1, n_tok, cdim)).sum(dim=-2)
        self.print(f"effective_in_contents.shape {effective_in_contents.shape}, {effective_in_contents.sum()}", channel="shapes")
        out_contents = batched_memories.matmul(effective_in_contents.view(*effective_in_contents.shape, -1)).squeeze()
        self.print(f"out_contents.shape {out_contents.shape}", channel="shapes")

        # POINTERS
        self.print(f"in_pointers.shape {in_pointers.shape}", channel="shapes")
        effective_in_pointers = (normed_weightings.view(bsize, n_mod, n_tok,    1) 
                                     * in_pointers.view(bsize,     1, n_tok, pdim)).sum(dim=-2)
        self.print(f"effective_in_pointers.shape {effective_in_pointers.shape}", channel="shapes")       
        out_pointers = T.zeros(bsize, n_mod, pdim)
        for module_idx in range(n_mod):
            routing_input = T.cat((effective_in_pointers[:, module_idx], out_contents[:, module_idx]), dim=-1)
            out_pointers[:, module_idx] = self.routing_transforms[module_idx](routing_input)
        self.print(f"out_pointers.shape {out_pointers.shape}", channel="shapes")
        
        # MEMORIZE
        out_weightings = F.relu(T.tanh(out_pointers.matmul(out_pointers.transpose(-2, -1)))).unsqueeze(-1)
        self.print(f"out_weightings.shape {out_weightings.shape}", channel="shapes")
        effective_out_contents = (out_weightings * out_contents.unsqueeze(1)).sum(dim=-2)
        self.print(f"effective_out_contents.shape {effective_out_contents.shape}", channel="shapes")
        normed_effective_in_contents = effective_in_contents/((effective_in_contents**2).sum(-1, keepdim=True)+1e-42)
        #normed_effective_in_contents = 1/((effective_in_contents)**2+(1e-21))
        perfect_memory = effective_out_contents.unsqueeze(-1).matmul(normed_effective_in_contents.unsqueeze(-2))
        self.print(f"perfect_memory.shape {perfect_memory.shape}", channel="shapes")
        mem_test = perfect_memory.matmul(effective_in_contents.unsqueeze(-1)).squeeze()
        self.print(f"mem_test.shape {mem_test.shape}, {mem_test.sum()}", channel="shapes")
        self.print(f"equality test\n{mem_test[0,0,:5].tolist()}\n{effective_out_contents[0,0,:5].tolist()}", channel="memtest")

        batched_memories = 0.5 * (batched_memories + perfect_memory)

        return out_pointers, out_contents, batched_memories
        
        
    def cycle(self, pointers, contents, n=0):
        self.print(f"INPUT: pointers.shape {pointers.shape} contents.shape {contents.shape}", channel="shapes")
        batched_memories = self.memories.unsqueeze(0).expand(pointers.shape[0], *self.memories.shape)
        self.print(f"batched_memories.shape {batched_memories.shape}")
        for cycle_idx in range(n or self.n_cycles):
            pointers, contents, batched_memories = self.pass_through_modules(pointers, contents, batched_memories)
            self.print(f"CYCLE RESULT {cycle_idx+1}: pointers.shape {pointers.shape} contents.shape {contents.shape}\n", channel="shapes")

        return pointers, contents, batched_memories


    def collect(self, receivers, pointers, contents):
        weightings = self.attend(receivers, pointers)
        self.print("activated weightings sum", weightings.sum(), channel="grad")
        effective_contents = (weightings * contents.unsqueeze(1)).sum(dim=-2)
        self.print("effective contents sum", effective_contents.sum(), channel="grad")
        return effective_contents
        

    def attend(self, receivers, pointers):
        raw_weightings = receivers.matmul(pointers.transpose(-2, -1))
        activated_weightings = F.relu(T.tanh(raw_weightings)).unsqueeze(-1)
        #self.print(f"raw_weightings {raw_weightings}")
        return activated_weightings


    def training_step(self, batch, batch_idx):
        inputs, _ = batch

        # PERCEIVE
        encoder_embeds = self.cnn_encoder(inputs)

        self.print("encoder_embeds.shape", encoder_embeds.shape, channel="shapes")
        in_pointers, in_contents = encoder_embeds.flatten(-2).transpose(-2,-1).split([self.pointer_dim, self.content_dim], dim=-1)

        # CYCLE
        out_pointers, out_contents, batched_memories = self.cycle(in_pointers, in_contents)
        #(out_pointers.sum()+out_contents.sum()+batched_memories.sum()).backward()

        # MATCH
        effective_returned_contents = self.collect(in_pointers, out_pointers, out_contents)
        self.print("effective_returned_contents.shape", effective_returned_contents.shape, channel="shapes")

        # RECONSTRUCT
        reshaped_returned_contents = effective_returned_contents.transpose(-2, -1).reshape(
            inputs.shape[0], self.content_dim, *encoder_embeds.shape[-2:])
        self.print(f"reshaped_returned_contents.shape {reshaped_returned_contents.shape}", channel="shapes")
        #imagination = self.cnn_decoder(reshaped_returned_contents)
        
        contents = encoder_embeds[:,self.pointer_dim:,:,:]
        self.print(f"contents.shape {contents.shape}", channel="shapes")
        imagination = T.sigmoid(self.cnn_decoder(contents))

        # ZERO GRAD
        optimizers = self.optimizers()
        #print("OPTIMIZERS", optimizers)
        for opt in optimizers:
            opt.zero_grad()

        # LOSS
        #recon_loss = F.mse_loss(effective_returned_contents, in_contents)
        recon_loss = F.mse_loss(imagination, inputs)

        # OPTIMIZE
        self.manual_backward(recon_loss)
        for opt in optimizers:
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
    resultpath = f"results/_latest/"
    os.makedirs(resultpath, exist_ok=True)
    with open(resultpath+"log.txt", "w") as file:
        file.write(argstr)

    # INIT ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-archive", action="store_true")
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
    parser.add_argument("--archive", type=str, default="")
    args = parser.parse_args()
    args.resultpath = resultpath
    args.logch = args.logch.split(",")
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
        os.makedirs(archivepath, exist_ok=True)
        for file in os.listdir(resultpath):
            shutil.copy(resultpath+file, archivepath+file)
            shutil.copy(os.curdir+"/"+__file__, archivepath+__file__)
