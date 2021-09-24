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

class AssociativeMemoryModule(pl.LightningModule):
    def __init__(self, content_dim, pointer_dim, mid_acti=nn.ReLU(), end_acti=nn.ReLU()):
        super.init()
        self.receiver = nn.Parameter(T.tensor(pointer_dim))
        init.kaiming_uniform_(self.receiver, a=math.sqrt(5))
        self.memory = nn.Linear(content_dim, content_dim, bias=False)
        self.pointer_transform = nn.Sequential(
            nn.Linear(content_dim+pointer_dim, pointer_dim),
            mid_acti,
            nn.Linear(pointer_dim, pointer_dim),
            end_acti
        )

    def forward(self, pointers, contents):
        weighting = F.relu(F.tanh(pointers.dot(self.receiver)))
        key = (weighting * contents).sum(dim=1)
        in_pointer = (weighting * pointers).sum(dim=1) 
        value = self.memory(key)


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
        """self.simple_eye = nn.Sequential(
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
        self.visual_imagination = nn.Sequential(
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

        self.simple_eye = nn.Sequential(
            nn.Conv2d(self.color_channels, 16, 3, 1, 0),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 0),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, pointer_dim+content_dim, 1, 1, 0),
            nn.ReLU()
        )
        self.visual_imagination = nn.Sequential(
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
            *[module.parameters() for module in [self.simple_eye, self.visual_imagination]]))
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


    def pass_trough_modules(self, in_pointers:T.Tensor, in_contents:T.Tensor, batched_memories:T.Tensor):
        batch_size = in_pointers.shape[0]
        self.print(f"batchsize {batch_size}")
        #in_pointers = in_pointers.transpose(-2, -1)

        # WEIGHTINGS
        self.print(f"in_pointers.shape {in_pointers.shape}", channel="shapes")
        raw_weightings = self.receivers.matmul(in_pointers.transpose(-2, -1))
        self.print("raw weights sum", raw_weightings.sum(), channel="grad")
        in_weightings = F.relu(T.tanh(raw_weightings)).unsqueeze(-1)
        self.print("activated weights sum", in_weightings.sum(), channel="grad")
        self.print(f"in_weightings.shape {in_weightings.shape}", channel="shapes")

        # CONTENTS
        effective_in_contents = (in_weightings * in_contents.unsqueeze(1)).sum(dim=-2)
        self.print(f"effective_in_contents.shape {effective_in_contents.shape}, {effective_in_contents.sum()}", channel="shapes")
        out_contents = batched_memories.matmul(effective_in_contents.unsqueeze(-1)).squeeze()
        self.print(f"out_contents.shape {out_contents.shape}", channel="shapes")

        # POINTERS
        self.print(f"in_pointers.unsqueeze(1).shape {in_pointers.unsqueeze(1).shape}", channel="shapes")
        effective_in_pointers = (in_weightings * in_pointers.unsqueeze(1)).sum(dim=-2)
        self.print(f"effective_in_pointers.shape {effective_in_pointers.shape}", channel="shapes")       
        out_pointers = T.zeros(batch_size, self.n_modules, self.pointer_dim)
        for module_idx in range(self.n_modules):
            routing_input = T.cat((effective_in_pointers[:, module_idx], effective_in_contents[:, module_idx]), dim=-1)
            out_pointers[:, module_idx] = self.routing_transforms[module_idx](routing_input)
        self.print(f"out_pointers.shape {out_pointers.shape}", channel="shapes")
        
        # MEMORIZE
        out_weightings = F.relu(T.tanh(out_pointers.matmul(out_pointers.transpose(-2, -1)))).unsqueeze(-1)
        self.print(f"out_weightings.shape {out_weightings.shape}", channel="shapes")
        effective_out_contents = (out_weightings * out_contents.unsqueeze(1)).sum(dim=-2)
        self.print(f"effective_out_contents.shape {effective_out_contents.shape}", channel="shapes")
        normed_effective_in_contents = effective_in_contents/((effective_in_contents**2).sum(-1, keepdim=True)+1e-42)
        perfect_memory = effective_out_contents.unsqueeze(-1).matmul(normed_effective_in_contents.unsqueeze(-2))
        self.print(f"perfect_memory.shape {perfect_memory.shape}", channel="shapes")
        mem_test = perfect_memory.matmul(effective_in_contents.unsqueeze(-1)).squeeze()
        self.print(f"mem_test.shape {mem_test.shape}, {mem_test.sum()}", channel="shapes")
        self.print(f"equality test\n{mem_test[0,0,:5].tolist()}\n{effective_out_contents[0,0,:5].tolist()}", channel="memtest")

        batched_memories = 0.5 * (batched_memories + perfect_memory)

        return out_pointers, out_contents, batched_memories
        
        
    def cycle(self, pointers, contents):
        self.print(f"INPUT: pointers.shape {pointers.shape} contents.shape {contents.shape}", channel="shapes")
        batched_memories = self.memories.unsqueeze(0).expand(pointers.shape[0], *self.memories.shape)
        self.print(f"batched_memories.shape {batched_memories.shape}")
        for cycle_idx in range(self.n_cycles):
            pointers, contents, batched_memories = self.pass_trough_modules(pointers, contents, batched_memories)
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
        eye_embeds = self.simple_eye(inputs)
        """
        self.print("eye_embeds.shape", eye_embeds.shape, channel="shapes")
        in_pointers, in_contents = eye_embeds.flatten(-2).transpose(-2,-1).split([self.pointer_dim, self.content_dim], dim=-1)

        # PREDICT
        out_pointers, out_contents, batched_memories = self.cycle(in_pointers, in_contents)
        #(out_pointers.sum()+out_contents.sum()+batched_memories.sum()).backward()

        # MATCH
        effective_returned_contents = self.collect(in_pointers, out_pointers, out_contents)
        self.print("effective_returned_contents.shape", effective_returned_contents.shape, channel="shapes")

        # RECONSTRUCT
        reshaped_returned_contents = effective_returned_contents.transpose(-2, -1).reshape(
            inputs.shape[0], self.content_dim, *eye_embeds.shape[-2:])
        self.print(f"reshaped_returned_contents.shape {reshaped_returned_contents.shape}", channel="shapes")
        #imagination = self.visual_imagination(reshaped_returned_contents)
        """
        eye_contents = eye_embeds[:,self.pointer_dim:,:,:]
        self.print(f"eye_contents.shape {eye_contents.shape}", channel="shapes")
        imagination = self.visual_imagination(eye_contents)

        # ZERO GRAD
        optimizers = self.optimizers()
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
            self.viz(inputs, imagination)


    def viz(self, *batch_arrays):
        rows_concat = np.concatenate([batch.detach().cpu().numpy() for batch in batch_arrays], axis=-2)
        column_concat = np.concatenate(rows_concat, axis=-1)
        result = column_concat.transpose(1,2,0)
        plt.imshow(result)
        plt.show()


    def get_dummy_set(self, size=1000, n_tokens=16):
        pointers = T.randn(size, n_tokens, self.pointer_dim)
        contents = T.randn(size, n_tokens, self.content_dim)
        dummy_set = pointers, contents
        return dummy_set
        
if __name__=="__main__":

    # INIT ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true")
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
    args = parser.parse_args()
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
