"""
Implement machine learning to classify gene biotypes into coding and non coding genes

Get coding and non coding genes from ncbi

Train the multilayer peceptrons to recornize coding/non coding patterns

evaluate on single cells transcripts

Dataset downloaded from ensemble database

27500 * 27500 coding and non coding rna of sequence sizes ranging from 100 to about 3200 bp

kmers extractions are performed not every bp position but every kmer length to cut down overlap!
cutting down overlaps, reduces vector sizes and hence memory.

Embedding the tokens helps to capture rna structure and similarities between kmers.
"""

import os
import numpy as np
from Bio import SeqIO
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
os.getcwd()
os.listdir()
os.listdir("data")
coding = "data/Mus_musculus.GRCm39.cds.all.fa"
non_coding = "data/Mus_musculus.GRCm39.ncrna.fa"


import numpy as np
from typing import List, Dict, Union
import math

# Kmer tokenizer
class KmerTonenizer:

    def __init__ (self, k = 9, add_special_tokens : bool = True):
        """
        K-mer tonenizer without overlaps for rna seqs. 
        args: 
            kmer for k
            special tokens like cls, sep, unk tokens
        """

        self.k = k
        self.add_special_tokens = add_special_tokens

        # Init vocab
        self.vocab = {}
        self.inverse_vocab = {}

        if self.add_special_tokens:
            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            for token in special_tokens:
                self._add_token(token)

    def _add_token(self, token: str):
        """ add a single token to the vocab"""

        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

    def build_vocab(self, sequence_cds: List[str], sequence_ncds: List[str]): #
        """
        build vocab from the list of sequences
        ARgs: sequences as strings
        """

        # generate all possible k-mers from the sequence
        all_kmers = set()

        for seq in sequence_cds:
            seq = seq.upper().strip()
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i + self.k]
                if all(base in "ACGTU" for base in kmer):
                    all_kmers.add(kmer)

        for seq in sequence_ncds:
            seq = seq.upper().strip()
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i + self.k]
                if all(base in "ACGTU" for base in kmer):
                    all_kmers.add(kmer)
        
        # add kmers to vocab
        for kmer in sorted(all_kmers):
            self._add_token(kmer)
        
        print(f"vocab size : {len(self.vocab)} tokens")

    def tokenize(self, sequences: str, add_special_tokens: bool = None, 
                 non_overlap_kmer_max_len : int = None, max_len: int = None) -> List[int]:
        """
        Convert a sequence into token ids
        input rna/dna
        return token ids
        """
        
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        
        #sequences = sequences.upper().strip()
        tokens = []

        # add cls token at the begining if requested
        if add_special_tokens:
            tokens.append(self.vocab["[CLS]"])

        # extract k-mers and convert to ids
        #for i in range(len(sequences) - self.k + 1): this is for overlapping kmer
        for i in range(0, len(sequences) - self.k + 1, self.k): # no overlaps
            kmer = sequences[i:i + self.k]

            # use unk for k-mers not in vocab
            if kmer in self.vocab:
                tokens.append(self.vocab[kmer])
            else:
                tokens.append(self.vocab.get("[UNK]", 1)) # default to 1 if unk exist

        if add_special_tokens:
            tokens.append(self.vocab["[SEP]"])

        # apply padding [PAD]
        if non_overlap_kmer_max_len is not None:
            pad_id = self.vocab.get("[PAD]", 0)
            #print(pad_id)
            if len(tokens) < non_overlap_kmer_max_len:
                # rigth padding
                tokens = tokens + [pad_id] * (non_overlap_kmer_max_len - len(tokens))
            else: tokens = tokens[:non_overlap_kmer_max_len] # truncate

        return tokens
    
    def sequences_to_ids(self, sequences: List[str], max_len : int = None,
                         non_overlap_kmer_max_len : int = None) -> List[List[int]]:
        """
        Convert multiple sequences to token ids
        """

        return [self.tokenize(seq, max_len = max_len, 
            non_overlap_kmer_max_len = non_overlap_kmer_max_len) for seq in sequences]
    
    def ids_to_sequences(self, token_ids: List[int]) -> str:
        """ Convert tokens back to sequences (approximate)"""
        tokens = [self.inverse_vocab.get(idx, "[UNK]") for idx in token_ids]
        #print(tokens)

        # filter out special tokens
        sequence_tokens = [token for token in tokens if not token.startswith("[")]
        #print(sequence_tokens)

        if not sequence_tokens:
            return ""
        
        # reconstruct sequences from k-mers
        sequence = sequence_tokens[0]
        for kmer in sequence_tokens[1:]:
            sequence += kmer[-1]

        return sequence
    
    @property
    def vocabulary_size(self) -> int:
        return(self.vocab)


### Now apply for database
fasta_cds_list = [str(record.seq) for record in SeqIO.parse(open(coding), "fasta")]
fasta_ncds_list = [str(record.seq) for record in SeqIO.parse(open(non_coding), "fasta")]

# rmove doublets
cds_unique = list(set(fasta_cds_list))
ncds_unique = list(set(fasta_ncds_list))

# sort by length
cds_sorted = sorted(cds_unique, key = len)
ncds_sorted = sorted(ncds_unique, key = len)

# extract rna seq lengths between > 100 and < 3500 for both conditions
cds_modicum = cds_sorted[25500:52500]
ncds_modicum = ncds_sorted[500:27500]
cds_modicum[1]

k = 9 # kmer size
#instantiate tokenizer
tokenizer = KmerTonenizer(k = k, add_special_tokens = True)
tokenizer.build_vocab(cds_modicum, ncds_modicum)
#max_len = max(len(seq) for seq in cds_modicum)
max_len = max(len(seq) for seq in ncds_modicum) 
non_overlap_kmer_max_len = math.ceil(max_len/k) + 1
tokens_cds = tokenizer.sequences_to_ids(cds_modicum, 
                            max_len, non_overlap_kmer_max_len)
tokens_cds[-1]
tokens_cds = np.array(tokens_cds)
tokens_cds.shape # without overlapping kmers, features are reduced to (27000, 348)
# We will se downstreams if they cut down the noise
memory_size = tokens_cds.nbytes / (1024**3)


tokens_ncds = tokenizer.sequences_to_ids(ncds_modicum, 
                        max_len, non_overlap_kmer_max_len)
tokens_ncds = np.array(tokens_ncds)
tokens_ncds.shape
memory_size = tokens_ncds.nbytes / (1024**3)

# Now add label and start working on model
labels_cds = np.ones(len(tokens_cds), dtype=np.int8)
labels_ncds = np.zeros(len(tokens_ncds), dtype = np.int8)

x = np.vstack([tokens_cds, tokens_ncds])
y = np.concatenate([labels_cds, labels_ncds])

# shuffle before triaing
from sklearn.utils import shuffle
x, y = shuffle(x, y, random_state = 42)
len(x), len(y)
x[1000:1020], y[1000:1020]
x.shape
# >>> x.shape
# (54000, 2940)

# >>> x[1000:1020], y[1000:1020]
# (array([[ 2, 36, 68, ...,  0,  0,  0],
#        [ 2, 19, 63, ...,  0,  0,  0],
#        [ 2, 60, 35, ...,  0,  0,  0],
#        ...,
#        [ 2, 19, 63, ...,  0,  0,  0],
#        [ 2, 19, 61, ...,  0,  0,  0],
#        [ 2, 61, 40, ...,  0,  0,  0]]), array([0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
#       dtype=int8))
# >>> 

### Buil pytorch model
import os
import torch
from torch import nn
import matplotlib.pyplot as plt

### pca data observation
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
plt.scatter(x_pca[ :, 0], x_pca[ :, 1], c = y, cmap = "coolwarm", alpha = 0.7)
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.title("pca projection of kmer tokens")
plt.colorbar(label = "class")
plt.show()

### tsne data observation
from sklearn.manifold import TSNE
x_tsne = TSNE(n_components=2, random_state=42).fit_transform(x)
plt.scatter(x_tsne[ :, 0], x_tsne[ :, 1], c = y, cmap = "coolwarm", alpha = 0.7)
plt.title("t-SNE projection of kmer tokens")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.colorbar(label = "class")
plt.show()


# Split dataset into training and spliting
y = torch.from_numpy(y).type(torch.float)
x = torch.from_numpy(x).type(torch.float)

# Setup device agnostic code
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else: device = torch.device("cpu")
print(f"Using device: {device}")

x, y = x.to(device), y.to(device)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

x_train.shape
x_test.shape
y_test.shape
y_train.shape

# biotype classification model 
class biotype_classification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, output, max_seq_len = None):
        """
        A linear and non linear model for biotype classification.
        kmer vocabs embedding included to help learn similarities kmers
        Args: 
            50 hidden layers
            embedding layers 64 (captures the kmer structure and similarities)
            output is 1 for a binary classification
            maximum length of the input sequences
            kmer vocabulary size 
        """

        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = 0
        )

        # positional embedding for sequence order
        self.max_seq_len = max_seq_len
        if max_seq_len:
            self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # get input dim for linear layers
        linear_input_size = embedding_dim

        self.stack_layer = nn.Sequential(
            nn.Linear(in_features = linear_input_size, out_features= hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3), #randomly turns off 30% neurons, force robost learing when training
            nn.Linear(in_features= hidden_units, out_features= hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features= hidden_units, out_features= output)
        )

    def forward(self, x):
        if x.shape != torch.long:
            x = x.long()
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        # for positional embedding
        if hasattr(self, "pos_embedding") and self.max_seq_len:
            positions = torch.arange(0, x.size(1), device = device).expand(x.size(0), x.size(1)).to(device)
            pos_embedded = self.pos_embedding(positions)
            embedded = embedded + pos_embedded

        mask = (x != 0).float().unsqueeze(-1) #mask non padding tokens
        embedded_mask = embedded * mask 
        pooled = embedded_mask.sum(dim = 1) / mask.sum(dim = 1) # average only non padding tokens
        return self.stack_layer(pooled)
    
### instantiate model
input = x_train.shape[1]
hidden_units = 512
output = 1
model = biotype_classification(
    vocab_size = len(tokenizer.vocabulary_size),
    embedding_dim = 64,
    hidden_units = 50,
    output = 1,
    max_seq_len = max_len)
model = model.to(device)

# First test for errors in eval mode before training
def batched_predict(model, data, batch_size = 32):
    model.eval()
    all_logits = []
    
    with torch.inference_mode():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch = batch.long().to(next(model.parameters()).device)
            logits = model(batch).squeeze()
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits)
logits = batched_predict(model, x_train, batch_size = 32)
pred = torch.round(torch.sigmoid(logits))
acc = (pred == y_train.cpu()).float().mean() 

## Define loss function
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1) # try aslo sgd

### Buil dataset/dataloader for training in batches
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(x_train.long(), y_train)
dataloader = DataLoader(dataset, batch_size = 128, shuffle = False)

## Build training
from timeit import default_timer as timer
from tqdm import tqdm

# training tinme func
def train_test_time (start: float, end: float) -> str:
    run_time = end - start
    return f"{run_time:.2f} seconds"

epochs = 3
model.train()
for epoch in tqdm(range(epochs)):
    start_time = timer()
    print(f"epoch: {epoch}\n---------------")

    train_loss, train_acc = 0, 0
    for batch, (x_batch, y_batch) in enumerate(dataloader):
        logits = model(x_batch).squeeze()
        pred = torch.round(torch.sigmoid(logits))

        loss = loss_fn(logits, y_batch)
        train_loss += loss

        acc = (pred == y_batch).float().mean()
        train_acc += acc
        acc = acc * 100

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 400 == 0:
            tqdm.write(f"Locked at {batch * len(x_batch)}/{len(dataloader.dataset)} samples")
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    end_time = timer()
    run_time = train_test_time(start_time, end_time)
    tqdm.write(f"Epoch : {epoch} | Loss : {loss:.4f} | Accuracy : {acc:.2f}% | Runtime : {run_time}")
        

# Test the model on test dataset
model.eval()
def batched_predict(model, data, batch_size = 32):
    model.eval()
    all_logits = []
    
    with torch.inference_mode():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch = batch.long().to(next(model.parameters()).device)
            logits = model(batch).squeeze()
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits)
logits = batched_predict(model, x_test, batch_size = 32)
pred = torch.round(torch.sigmoid(logits))
acc = (pred == y_test.cpu()).float().mean() * 100
acc 
pred[0:30]
y_test[0:30]

# >>> acc
# tensor(98.2315)
# >>> pred[0:30]
# tensor([0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
#         1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0.])
# >>> y_test[0:30]
# tensor([0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
#         1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0.], device='mps:0')


import gc
torch.mps.empty_cache()
gc.collect()


####  One hot encoding  ########
# bases = ["A", "C", "G", "T"]
# base_to_index = {b: i for i, b in enumerate(bases)}

# def one_hot_encoding(seq):
#     """
#     Here, one hode codes the rna to base binary

#     shape is (seq_len, 4)
#     """

#     one_hot = np.zeros((len(seq), 4), dtype = np.float32)
#     for i, base in enumerate(seq):
#         if base in base_to_index:
#             one_hot[i, base_to_index[base]] = 1 # else leave the others as 0s
#     return one_hot
    
# ######################################
# # Coding protein encoding
# ######################################

# fasta_cds = SeqIO.parse(open(coding), "fasta") 
# encoded_sequences = []
# seq_ids = []
# for record in fasta_cds:
#     seq = str(record.seq)
#     one_hot_en = one_hot_encoding(seq)
#     match = re.search(r'gene_symbol:([A-Za-z0-9_.-]+)', record.description)
#     if match:
#         encoded_sequences.append(one_hot_en)
#         gene_symbol = match.group(1)
#         seq_ids.append(gene_symbol)
#         print(f"Encoded for gene : {gene_symbol}")

# len(seq_ids) # list contains isoforms. total tO ~66K genes
# len(encoded_sequences)
# encoded_sequences[0].shape

# # gene has variable lenths so it is important to pad them to the same len
# max_len = max(seq.shape[0] for seq in encoded_sequences)

# #pad sequence to the same len
# flat_seqs = [seq.flatten() for seq in encoded_sequences]
# padded_flat = pad_sequences(flat_seqs, maxlen = max_len*4, dtype = "float32", padding = "post")
# padded = padded_flat.reshape(len(encoded_sequences), max_len, 4)
# padded.shape
# padded[0] # the padded spaces takes 0 all along

# ######################################
# # Non coding protein encoding
# ######################################

# fasta_ncds = SeqIO.parse(open(non_coding), "fasta")
# nc_encoded_seq = []
# nc_seq_ids = []
# for record in fasta_ncds:
#     seq = str(record.seq)
#     one_hot_en = one_hot_encoding(seq)
#     match = re.search(r"gene_symbol:([A-Za-z0-9_.-]+)", record.description)
#     if match:
#         nc_encoded_seq.append(one_hot_en)
#         gene_symbol = match.group(1)
#         nc_seq_ids.append(gene_symbol)
#         print(f"Encoded for gene : {gene_symbol}")

# len(nc_seq_ids)
# len(nc_encoded_seq)
# nc_encoded_seq[0]

# # discard long genes to limit memory use
# # for seq in 

# # padding
# max_len = max(seq.shape[0] for seq in nc_encoded_seq)

# flat_seqs = [seq.flatten() for seq in nc_encoded_seq]
# padded_flat = pad_sequences(flat_seqs, maxlen = max_len*4, dtype = "float32", padding = "post")
# nc_padded = padded_flat.reshape(len(nc_encoded_seq), max_len, 4)

# len(nc_padded)
# nc_padded.shape
# len(nc_seq_ids)
# nc_padded[0]

# memory_size = nc_padded.nbytes / (1024**3)  # Size in GB
# print(f"Data size in memory: {memory_size:.2f} GB")
# #Data size in memory: 45.76 GB

# # add label
# y = np.zeros(nc_padded.shape[0])
# y.shape


# ################ rna structural feature extraction 
# def extract_rna_features(sequence):
#     """Extract biological features instead of raw sequence"""
#     features = {
#         'length': len(sequence),
#         'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence),
#         'a_ratio': sequence.count('A') / len(sequence),
#         'c_ratio': sequence.count('C') / len(sequence),
#         'g_ratio': sequence.count('G') / len(sequence),
#         't_ratio': sequence.count('T') / len(sequence),
#     }
#     return list(features.values())

# features = np.array([extract_rna_features(record.seq) for record in fasta_ncds])
# print(f"Shape: {features.shape}")  
# features[29389]