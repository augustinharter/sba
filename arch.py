import numpy as np
import torch as T
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math 
from itertools import chain
import argparse
import wandb

class AssociativeMemoryModule(nn.Module):
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


class SemanticBusArchitecture(nn.Module):
    def __init__(self, n_modules, pointer_dim, content_dim, n_cycles=1, log_channels=["loss"]):
        super().__init__()
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        self.log_channels = log_channels
        self.n_modules = n_modules
        self.n_cycles = n_cycles
        self.content_dim = content_dim
        self.pointer_dim = pointer_dim

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

        self.reset_parameters()
        self.configure_optimizers()


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.receivers, a=math.sqrt(5))
        init.kaiming_uniform_(self.memories, a=math.sqrt(5))
        self.log("memories init:", self.memories, channel="init")


    def configure_optimizers(self):
        self.routing_transforms_optimizer = T.optim.Adam(chain(
            *[module.parameters() for module in self.routing_transforms]))
        self.memories_optimizer = T.optim.Adam([self.memories])
        self.receivers_optimizer = T.optim.Adam([self.receivers])


    def step_all_optimizers(self):
        self.routing_transforms_optimizer.step()
        self.memories_optimizer.step()
        self.receivers_optimizer.step()


    def zero_all_optimizers(self):
        self.routing_transforms_optimizer.zero_grad()
        self.memories_optimizer.zero_grad()
        self.receivers_optimizer.zero_grad()


    def log(self, *args, channel="debug"):
        if channel in self.log_channels:
            print(*args)


    def cycle(self, in_pointers:T.Tensor, in_contents:T.Tensor, batched_memories:T.Tensor):
        batch_size = in_pointers.shape[0]
        self.log(f"batchsize {batch_size}")
        #in_pointers = in_pointers.transpose(-2, -1)

        # WEIGHTINGS
        self.log(f"in_pointers.shape {in_pointers.shape}", channel="shapes")
        raw_weightings = self.receivers.matmul(in_pointers.transpose(-2, -1))
        self.log("raw weights sum", raw_weightings.sum(), channel="grad")
        in_weightings = F.relu(T.tanh(raw_weightings)).unsqueeze(-1)
        self.log("activated weights sum", in_weightings.sum(), channel="grad")
        self.log(f"in_weightings.shape {in_weightings.shape}", channel="shapes")

        # CONTENTS
        effective_in_contents = (in_weightings * in_contents.unsqueeze(1)).sum(dim=-2)
        self.log(f"effective_in_contents.shape {effective_in_contents.shape}, {effective_in_contents.sum()}", channel="shapes")
        out_contents = batched_memories.matmul(effective_in_contents.unsqueeze(-1)).squeeze()
        self.log(f"out_contents.shape {out_contents.shape}", channel="shapes")

        # POINTERS
        self.log(f"in_pointers.unsqueeze(1).shape {in_pointers.unsqueeze(1).shape}", channel="shapes")
        effective_in_pointers = (in_weightings * in_pointers.unsqueeze(1)).sum(dim=-2)
        self.log(f"effective_in_pointers.shape {effective_in_pointers.shape}", channel="shapes")       
        out_pointers = T.zeros(batch_size, self.n_modules, self.pointer_dim)
        for module_idx in range(self.n_modules):
            routing_input = T.cat((effective_in_pointers[:, module_idx], effective_in_contents[:, module_idx]), dim=-1)
            out_pointers[:, module_idx] = self.routing_transforms[module_idx](routing_input)
        self.log(f"out_pointers.shape {out_pointers.shape}", channel="shapes")
        
        # MEMORIZE
        out_weightings = F.relu(T.tanh(out_pointers.matmul(out_pointers.transpose(-2, -1)))).unsqueeze(-1)
        self.log(f"out_weightings.shape {out_weightings.shape}", channel="shapes")
        effective_out_contents = (out_weightings * out_contents.unsqueeze(1)).sum(dim=-2)
        self.log(f"effective_out_contents.shape {effective_out_contents.shape}", channel="shapes")
        normed_effective_in_contents = effective_in_contents/((effective_in_contents**2).sum(-1, keepdim=True)+1e-42)
        perfect_memory = effective_out_contents.unsqueeze(-1).matmul(normed_effective_in_contents.unsqueeze(-2))
        self.log(f"perfect_memory.shape {perfect_memory.shape}", channel="shapes")
        mem_test = perfect_memory.matmul(effective_in_contents.unsqueeze(-1)).squeeze()
        self.log(f"mem_test.shape {mem_test.shape}, {mem_test.sum()}", channel="shapes")
        self.log(f"equality test\n{mem_test[0,0,:5].tolist()}\n{effective_out_contents[0,0,:5].tolist()}", channel="memtest")

        batched_memories = 0.5 * (batched_memories + perfect_memory)

        return out_pointers, out_contents, batched_memories
        
        
    def forward(self, pointers, contents):
        self.log(f"INPUT: pointers.shape {pointers.shape} contents.shape {contents.shape}")
        batched_memories = self.memories.unsqueeze(0).expand(pointers.shape[0], *self.memories.shape)
        self.log(f"batched_memories.shape {batched_memories.shape}")
        for cycle_idx in range(self.n_cycles):
            pointers, contents, batched_memories = self.cycle(pointers, contents, batched_memories)
            self.log(f"CYCLE RESULT {cycle_idx+1}: pointers.shape {pointers.shape} contents.shape {contents.shape}\n")

        return pointers, contents, batched_memories


    def collect(self, receivers, pointers, contents):
        weightings = self.attend(receivers, pointers)
        self.log("activated weightings sum", weightings.sum(), channel="grad")
        effective_contents = (weightings * contents.unsqueeze(1)).sum(dim=-2)
        self.log("effective contents sum", effective_contents.sum(), channel="grad")
        return effective_contents
        

    def attend(self, receivers, pointers):
        raw_weightings = receivers.matmul(pointers.transpose(-2, -1))
        activated_weightings = F.relu(T.tanh(raw_weightings)).unsqueeze(-1)
        #self.log(f"raw_weightings {raw_weightings}")
        return activated_weightings


    def train_step(self, batch):
        in_pointers, in_contents = batch
        in_pointers = in_pointers.to(self.device)
        in_contents = in_contents.to(self.device)

        out_pointers, out_contents, batched_memories = self.forward(in_pointers, in_contents)
        #(out_pointers.sum()+out_contents.sum()+batched_memories.sum()).backward()

        effective_returned_contents = self.collect(in_pointers, out_pointers, out_contents)

        # STEPPING
        self.zero_all_optimizers()
        recon_loss = F.mse_loss(effective_returned_contents, in_contents)
        recon_loss.backward()
        tmp = self.receivers.clone()
        self.step_all_optimizers()
        self.log("receivers changed", (tmp-self.receivers).sum(), channel="grad")

        # LOGGING
        loss = recon_loss.detach().cpu().item()
        self.log(f"GRAD self.receivers.grad.sum() {self.receivers.grad.sum()}", channel="grad-main")
        self.log(f"LOSS {loss}", channel="loss")
        wandb.log({"loss": loss})
        

    def dummy_batch(self, n_steps=1, batchsize=16, n_tokens=16):
        pointers = T.randn(batchsize, n_tokens, self.pointer_dim)
        contents = T.randn(batchsize, n_tokens, self.content_dim)
        batch = pointers, contents

        for _ in range(n_steps):
            self.train_step(batch)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logch", type=str, default="loss")
    parser.add_argument("--dummy-steps", type=int, default=100)
    parser.add_argument("--dummy-inputs", type=int, default=32)
    parser.add_argument("--pointer-dims", type=int, default=16)
    parser.add_argument("--content-dims", type=int, default=32)
    parser.add_argument("--modules", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()
    args.logch = args.logch.split(",")
    print(args.__dict__)

    # WANDB
    wandb.init(project='sba', entity='aharter', config=args.__dict__, notes=args.notes)
    config = wandb.config

    sba = SemanticBusArchitecture(
        args.modules, 
        args.pointer_dims, 
        args.content_dims, 
        log_channels=args.logch, 
        n_cycles=args.cycles)
    sba.dummy_batch(n_steps=args.dummy_steps, batchsize=args.batchsize, n_tokens=args.dummy_inputs)
