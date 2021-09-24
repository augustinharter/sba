import torch as T
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math

# SETUP
n = 1000
key_dim = 128
value_dim = 128
s = 16
sparse = False

# CLASSIC
if not sparse:
    keys = T.randn(n, key_dim, dtype=T.float32)
    keys = keys/keys.norm(dim=-1, keepdim=True)
    values = T.randn(n, value_dim, dtype=T.float32)
    values = values/values.norm(dim=-1, keepdim=True)
else:
    keys = T.zeros(n, key_dim, dtype=T.float32)
    keys[T.arange(n)[:,None], T.randint(key_dim, (n,s))] = 1
    values = T.zeros(n, value_dim, dtype=T.float32)
    values[T.arange(n)[:,None], T.randint(value_dim, (n,s))] = 1

print(keys)

# WRITE
mem = keys.view(n, key_dim, 1).matmul(values.view(n, 1, value_dim)).mean(dim=0)

# READ
read = keys.matmul(mem)

# COMPARE
base_diff = T.relu(T.cosine_similarity(values.tile(n,1,1), values.tile(n,1,1).transpose(0,1), dim=-1))
read_diff = T.relu(T.cosine_similarity(values.tile(n,1,1), read.tile(n,1,1).transpose(0,1), dim=-1))
print("trace mean", base_diff.trace()/n/(base_diff).mean(),
                    read_diff.trace()/n/(read_diff).mean())
plt.subplot(1,2,1)
plt.imshow(base_diff)
plt.subplot(1,2,2)
plt.imshow(read_diff)
plt.show()
