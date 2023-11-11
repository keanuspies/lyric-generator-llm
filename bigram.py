import re
import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# hyper params
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
# ------------
torch.manual_seed(1337)

# get dataset
filenames = glob.glob('data/*txt')
with open('data/all_lyrics.txt', 'w') as outfile:
    for fname in filenames:
        if 'all_lyrics' in fname: continue
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

with open('data/all_lyrics.txt', 'r') as f:
    text = f.read()

with open('data/all_lyrics.txt', 'w') as f:
    list_of_chars = "[^\n\"$&\'(),-./0123456789:?ABCDEFGHIJKLMNOPQRSTUVWY\[\]abcdefghijklmnopqrstuvwxyz ]"
    regex = re.compile(list_of_chars)
    f.write(regex.sub('', text))

with open('data/all_lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# -----------------

# info about data
chars = sorted(list(set(text)))
vocab_size = len(chars)
# encoder and decoder
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# data to torch and splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of inputs and targets
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # estimate loss over many batches
        # this makes the loss less noisy
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next torkne from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx: (B, T) 
        logits = self.token_embedding_table(idx) # (B, T = 8, C = channels == vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T = 8 for the 8 context) indices of current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # now (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
# define model and optimizer
model  = BigramLanguageModel(vocab_size)
m = model.to(device)

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train
for steps in range(max_iters): 

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss:{losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # sample new batch of data 
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
