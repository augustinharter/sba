import contextlib
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
from data import CIFAR10DataModule, MNISTDataModule
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os, sys
from datetime import datetime


class SemanticBusArchitecture(pl.LightningModule):
    def __init__(self, n_modules, pointer_dim, content_dim, 
        n_cycles=1, log_channels=[], dyn_rec=True, dbl_content=False, 
        use_module_mem=False, args=None):
        super().__init__()

        # CONSTANTS & FLAGS & ARGUMENTS
        self.args = args
        self.dyn_rec = dyn_rec
        self.dbl_content = dbl_content
        self.automatic_optimization = False
        self.print_channels = log_channels
        self.n_modules = n_modules
        self.n_cycles = n_cycles
        self.content_dim = content_dim
        self.pointer_dim = pointer_dim
        self.use_module_mem = use_module_mem
        self.lr = args.lr
        self.losslog = []

        # ENCODER DECODER
        self.utility_parameters = []
        if args.dataset=="mnist":
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 0),
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
            if self.args.classify:
                self.classifier = nn.Linear(content_dim, 10)
                self.result_receiver = nn.Parameter(T.randn(pointer_dim))
                self.utility_parameters.extend([self.classifier.parameters(), [self.result_receiver]])
            if args.reconstruct:
                self.utility_parameters.extend([self.cnn_decoder.parameters(), self.cnn_encoder.parameters()])
        elif args.dataset=="cifar10":
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 16, 3, 1, 0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, pointer_dim+content_dim, 1, 1, 0),
                nn.ReLU()
            )
            self.utility_parameters.append(self.cnn_encoder.parameters())
            if self.args.reconstruct:
                self.cnn_decoder = nn.Sequential()
                self.utility_parameters.append(self.cnn_decoder.parameters())
            if self.args.classify:
                self.classifier = nn.Linear(content_dim, 10)
                self.result_receiver = nn.Parameter(T.randn(pointer_dim))
                self.utility_parameters.extend([self.classifier.parameters(), [self.result_receiver]])

        # MODULES
        self.receivers = nn.Parameter(T.zeros(n_modules, pointer_dim))
        self.module_mem = nn.Parameter(T.zeros(n_modules, content_dim, content_dim))
        self.routing_transforms = [nn.Sequential(
            nn.Linear((1+dbl_content)*content_dim+pointer_dim, pointer_dim),
            nn.ReLU(),
            nn.Linear(pointer_dim, (1+dyn_rec)*pointer_dim),
            nn.ReLU()
        ) for _ in range(n_modules)]


        # INIT PARAMETERS AND OPTIMIZERS
        self.reset_parameters()
        self.configure_optimizers()


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.receivers, a=math.sqrt(5))
        init.kaiming_uniform_(self.module_mem, a=math.sqrt(5))
        #init.kaiming_uniform_(self.result_receiver, a=math.sqrt(5))
        self.print("memories init:", self.module_mem, channel="init")


    def configure_optimizers(self):
        routing_transforms_optimizer = T.optim.Adam(chain(
            *[module.parameters() for module in self.routing_transforms]), lr=self.lr)
        utility_optimizer = T.optim.Adam(chain(*self.utility_parameters), lr=self.lr)
        module_memories_optimizer = T.optim.Adam([self.module_mem], lr=self.lr)
        module_receivers_optimizer = T.optim.Adam([self.receivers], lr=self.lr)
        return routing_transforms_optimizer, module_memories_optimizer, module_receivers_optimizer, utility_optimizer


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


    def pass_through_modules(self, in_pointers:T.Tensor, in_contents:T.Tensor, 
                            batched_module_mems:T.Tensor, batched_receivers:T.Tensor):
        bsize, n_tok, pdim = in_pointers.shape
        n_mod = batched_receivers.shape[1]
        cdim = in_contents.shape[-1]

        # ATTENTION WEIGHTING
        normed_weightings = self.calc_attention_weightings(batched_receivers, in_pointers)

        # CONTENTS
        effective_in_contents = self.weight_and_sum(normed_weightings, in_contents)
        out_contents = batched_module_mems.matmul(effective_in_contents.view(*effective_in_contents.shape, -1)).squeeze()

        # POINTERS
        effective_in_pointers = self.weight_and_sum(normed_weightings, in_pointers)
        out_pointers = T.zeros(bsize, n_mod, pdim * (self.dyn_rec+1))
        for module_idx in range(n_mod):
            routing_input = T.cat((effective_in_pointers[:, module_idx], out_contents[:, module_idx]), dim=-1)
            out_pointers[:, module_idx] = self.routing_transforms[module_idx](routing_input)
        
        # RECEIVER UPDATE
        if self.dyn_rec:
            out_pointers, receiver_deltas = out_pointers.split(self.pointer_dim, dim=-1)
            batched_receivers = batched_receivers + receiver_deltas
        
        # MEMORIZE
        if self.use_module_mem:
            out_weightings = F.relu(T.tanh(out_pointers.matmul(out_pointers.transpose(-2, -1)))).unsqueeze(-1)
            effective_out_contents = (out_weightings * out_contents.unsqueeze(1)).sum(dim=-2)
            inverted_effective_in_contents = effective_in_contents/((effective_in_contents**2).sum(-1, keepdim=True)+1e-42)
            #inverted_effective_in_contents = 1/((effective_in_contents)**2+(1e-21))
            perfect_memory = effective_out_contents.unsqueeze(-1).matmul(inverted_effective_in_contents.unsqueeze(-2))
            batched_module_mems = 0.5 * (batched_module_mems + perfect_memory)
        
        # PRINTS
        if self.print_channels:
            print("PRINTING", self.print_channels)
            self.print(f"in_pointers.shape {in_pointers.shape}", channel="shapes")
            self.print(f"batched_receivers.shape {batched_receivers.shape}", channel="shapes")
            #self.print(f"raw_weightings.shape {raw_weightings.shape}", channel="shapes")
            #self.print("raw weights sum", raw_weightings.sum(), channel="grad")
            #self.print("activated weights sum", activated_weightings.sum(), channel="grad")
            #self.print(f"activated_weightings.shape {activated_weightings.shape}", channel="shapes")
            self.print("normed_weightings.shape", normed_weightings.shape, normed_weightings.sum(0).sum(-1), channel="shapes")
            self.print(f"batchsize {bsize}", channel="shapes")
            self.print(f"out_contents.shape {out_contents.shape}", channel="shapes")
            self.print(f"effective_in_contents.shape {effective_in_contents.shape}, {effective_in_contents.sum()}", channel="shapes")
            self.print(f"in_pointers.shape {in_pointers.shape}", channel="shapes")
            self.print(f"effective_in_pointers.shape {effective_in_pointers.shape}", channel="shapes")       
            self.print(f"out_pointers.shape {out_pointers.shape}", channel="shapes")
            #self.print(f"out_weightings.shape {out_weightings.shape}", channel="shapes")
            self.print(f"effective_out_contents.shape {effective_out_contents.shape}", channel="shapes")
            if self.use_module_mem:
                self.print(f"perfect_memory.shape {perfect_memory.shape}", channel="shapes")
                mem_test = perfect_memory.matmul(effective_in_contents.unsqueeze(-1)).squeeze()
                self.print(f"mem_test.shape {mem_test.shape}, {mem_test.sum()}", channel="shapes")
            self.print(f"equality test\n{mem_test[0,0,:5].tolist()}\n{effective_out_contents[0,0,:5].tolist()}", channel="memtest")

        return out_pointers, out_contents, batched_module_mems, batched_receivers
        
        
    def cycle(self, pointers, contents, n=0):
        bsize = pointers.shape[0]
        batched_module_mems = self.module_mem.unsqueeze(0).expand(bsize, *self.module_mem.shape)
        batched_receivers = self.receivers.unsqueeze(0).expand(bsize, *self.receivers.shape)

        for cycle_idx in range(n or self.n_cycles):
            pointers, contents, batched_module_mems, batched_receivers = self.pass_through_modules(
                pointers, contents, batched_module_mems, batched_receivers)

        return pointers, contents, batched_module_mems, batched_receivers


    def weight_and_sum(self, weightings, vectors):
        n_mod = weightings.shape[1]
        bsize, n_tok, cdim = vectors.shape
        effective_vectors = (weightings.view(bsize, n_mod, n_tok,    1) 
                              * vectors.view(bsize,     1, n_tok, cdim)).sum(dim=-2)

        if self.print_channels:
            self.print("effective vectors sum", effective_vectors.sum(), channel="grad")
        return effective_vectors


    def calc_attention_weightings(self, receivers, pointers, norm="modules"):
        bsize, n_mod, pdim = receivers.shape
        n_tok = pointers.shape[1]

        #TODO try squashing function instead
        raw_weightings = F.cosine_similarity(receivers.view(bsize, n_mod,     1, pdim), 
                                              pointers.view(bsize,     1, n_tok, pdim), dim=-1)
        
        activated_weightings = (raw_weightings.abs())

        # NORM
        if norm=="modules":
            normed_weightings = activated_weightings/activated_weightings.sum(dim=1, keepdim=True)
        else:
            exit(1)

        if self.print_channels:
            self.print("normed_weightings sum", normed_weightings.sum(), channel="grad")

        # OLD
        #raw_weightings = receivers.matmul(in_pointers.transpose(-2, -1))
        #raw_weightings = receivers.matmul(pointers.transpose(-2, -1))
        #activated_weightings = F.relu(T.tanh(raw_weightings)).unsqueeze(-1)
        #self.print(f"raw_weightings {raw_weightings}")
        return normed_weightings


    def attend_and_collect(self, receivers, pointers, contents):
        weightings = self.calc_attention_weightings(receivers, pointers)
        effective_contents = self.weight_and_sum(weightings, contents)
        
        if self.print_channels:
            self.print("weightings sum", weightings.sum(), channel="grad")
            self.print("effective contents sum", effective_contents.sum(), channel="grad")
        # OLD
        #effective_contents = (weightings * contents.unsqueeze(1)).sum(dim=-2)
        return effective_contents


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        bsize = inputs.shape[0]
        combined_loss = 0

        # PERCEIVE
        encoder_embeds = self.cnn_encoder(inputs)

        self.print("encoder_embeds.shape", encoder_embeds.shape, channel="shapes")
        in_pointers, in_contents = encoder_embeds.flatten(-2).transpose(-2,-1).split([self.pointer_dim, self.content_dim], dim=-1)

        # CYCLE
        out_pointers, out_contents, batched_module_mems, batched_receivers = self.cycle(in_pointers, in_contents)
        #(out_pointers.sum()+out_contents.sum()+batched_module_mems.sum()).backward()

        # RECONSTRUCT
        if self.args.reconstruct:
            effective_returned_contents = self.attend_and_collect(in_pointers, out_pointers, out_contents)
            self.print("effective_returned_contents.shape", effective_returned_contents.shape, channel="shapes")
            reshaped_returned_contents = effective_returned_contents.transpose(-2, -1).reshape(
                inputs.shape[0], self.content_dim, *encoder_embeds.shape[-2:])
            self.print(f"reshaped_returned_contents.shape {reshaped_returned_contents.shape}", channel="shapes")
            #imagination = self.cnn_decoder(reshaped_returned_contents)
        
            contents = encoder_embeds[:,self.pointer_dim:,:,:]
            self.print(f"contents.shape {contents.shape}", channel="shapes")
            reconstruction = T.sigmoid(self.cnn_decoder(contents))
            recon_loss = F.mse_loss(reconstruction, inputs)
            combined_loss = combined_loss + recon_loss

        # CLASSIFY
        if self.args.classify:
            batched_result_receiver = self.result_receiver.unsqueeze(0).unsqueeze(0).expand(bsize, 1, self.pointer_dim)
            effective_returned_contents = self.attend_and_collect(batched_result_receiver, 
                                                                  out_pointers, out_contents)
            classification = self.classifier(effective_returned_contents).squeeze()
            #F.cross_entropy(T.randn(5,3), T.randint(3, (5,)))
            classification_loss = F.cross_entropy(classification, labels)
            combined_loss = combined_loss + classification_loss

        # OPTIMIZE
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()
        self.manual_backward(combined_loss)
        for opt in optimizers:
            opt.step()

        # LOGGING
        self.print(f"LOSS {combined_loss}", channel="loss")
        self.log("loss", combined_loss.item())
        #self.print(f"GRAD self.receivers.grad.sum() {self.receivers.grad.sum()}", channel="grad-main")
        #self.print(f"INPUT VALUE RANGE {inputs.min(), inputs.max()}", channel="values")

        #VIZ
        if not batch_idx%args.printevery:
            if self.args.reconstruct:
                self.viz(batch_idx, inputs, reconstruction)
            if self.args.classify:
                accuracy = (classification.argmax(dim=-1)==labels).float().mean()
                #print("effective_returned_contents.sum()", effective_returned_contents.sum())                                                    
                print("classification", classification.argmax(dim=-1))
                print(f"accuracy {accuracy}")
                self.log("acc", accuracy)


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
    parser.add_argument("-classify", action="store_true")
    parser.add_argument("-reconstruct", action="store_true")
    parser.add_argument("-dynrec", action="store_true")
    parser.add_argument("--logch", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--pointer-dims", type=int, default=16)
    parser.add_argument("--content-dims", type=int, default=32)
    parser.add_argument("--modules", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--nThreads", type=int, default=1)
    parser.add_argument("--printevery", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--archive", type=str, default="")
    args = parser.parse_args()
    args.resultpath = resultpath
    args.logch = args.logch.split(",") if args.logch else []
    print("ARGS", args.__dict__)
    T.manual_seed(args.seed)

    # INIT LOGGER
    wandblogger = WandbLogger(project='sba', entity='aharter', notes=args.notes, config=args.__dict__, save_dir="saves")

    # INIT DATA
    if args.dataset=="mnist":
       dm = MNISTDataModule("../data", batch_size=args.batchsize)
    if args.dataset=="cifar10":
       dm = CIFAR10DataModule("../data", batch_size=args.batchsize)

    # INIT MODEL
    sba = SemanticBusArchitecture(
        args.modules, 
        args.pointer_dims, 
        args.content_dims, 
        log_channels=args.logch, 
        n_cycles=args.cycles,
        dyn_rec=args.dynrec,
        args=args
    )

    # OVERFIT
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
