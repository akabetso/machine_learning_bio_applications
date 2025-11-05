"""
Implement deep learning to classify gene biotypes into coding and non coding genes

Get coding and non coding genes from ncbi

Train the multilayer peceptrons to recornize coding/non coding patterns

evaluate on single cells transcripts

Dataset downloaded from ensemble database
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

bases = ["A", "C", "G", "T"]
base_to_index = {b: i for i, b in enumerate(bases)}

def one_hot_encoding(seq):
    """
    Here, one hode codes the rna to base binary

    shape is (seq_len, 4)
    """

    one_hot = np.zeros((len(seq), 4), dtype = np.float32)
    for i, base in enumerate(seq):
        if base in base_to_index:
            one_hot[i, base_to_index[base]] = 1 # else leave the others as 0s
    return one_hot
    
######################################
# Coding protein encoding
######################################

fasta_cds = SeqIO.parse(open(coding), "fasta") 
encoded_sequences = []
seq_ids = []
for record in fasta_cds:
    seq = str(record.seq)
    one_hot_en = one_hot_encoding(seq)
    match = re.search(r'gene_symbol:([A-Za-z0-9_.-]+)', record.description)
    if match:
        encoded_sequences.append(one_hot_en)
        gene_symbol = match.group(1)
        seq_ids.append(gene_symbol)
        print(f"Encoded for gene : {gene_symbol}")

len(seq_ids) # list contains isoforms. total tO ~66K genes
len(encoded_sequences)
encoded_sequences[0].shape

# gene has variable lenths so it is important to pad them to the same len
max_len = max(seq.shape[0] for seq in encoded_sequences)

#pad sequence to the same len
flat_seqs = [seq.flatten() for seq in encoded_sequences]
padded_flat = pad_sequences(flat_seqs, maxlen = max_len*4, dtype = "float32", padding = "post")
padded = padded_flat.reshape(len(encoded_sequences), max_len, 4)
padded.shape
padded[0] # the padded spaces takes 0 all along

######################################
# Non coding protein encoding
######################################

fasta_ncds = SeqIO.parse(open(non_coding), "fasta")
nc_encoded_seq = []
nc_seq_ids = []
for record in fasta_ncds:
    seq = str(record.seq)
    one_hot_en = one_hot_encoding(seq)
    match = re.search(r"gene_symbol:([A-Za-z0-9_.-]+)", record.description)
    if match:
        nc_encoded_seq.append(one_hot_en)
        gene_symbol = match.group(1)
        nc_seq_ids.append(gene_symbol)
        print(f"Encoded for gene : {gene_symbol}")

len(nc_seq_ids)
len(nc_encoded_seq)
nc_encoded_seq[0]

# discard long genes to limit memory use
# for seq in 

# padding
max_len = max(seq.shape[0] for seq in nc_encoded_seq)

flat_seqs = [seq.flatten() for seq in nc_encoded_seq]
padded_flat = pad_sequences(flat_seqs, maxlen = max_len*4, dtype = "float32", padding = "post")
nc_padded = padded_flat.reshape(len(nc_encoded_seq), max_len, 4)

len(nc_padded)
nc_padded.shape
len(nc_seq_ids)
nc_padded[0]

memory_size = nc_padded.nbytes / (1024**3)  # Size in GB
print(f"Data size in memory: {memory_size:.2f} GB")
#Data size in memory: 45.76 GB

# add label
y = np.zeros(nc_padded.shape[0])
y.shape



# #######
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


# K-mer tokenizer
"""
fit in something here: 
"""

import numpy as np
from typing import List, Dict, Union

class KmerTonenizer:
    """
    Here, a simple k-mer tonenizer for rna seqs
    """

    def __init__ (self, k = 3, add_special_tokens : bool = True):
        """
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

    def build_vocab(self, sequence_cds: List[str], sequence_ncds: List[str]):
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

    def tokenize(self, sequences: str, add_special_tokens: bool = None, max_len: int = None) -> List[int]:
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
        for i in range(len(sequences) - self.k + 1):
            kmer = sequences[i:i + self.k]

            # use unk for k-mers not in vocab
            if kmer in self.vocab:
                tokens.append(self.vocab[kmer])
            else:
                tokens.append(self.vocab.get("[UNK]", 1)) # default to 1 if unk exist

        if add_special_tokens:
            tokens.append(self.vocab["[SEP]"])

        # apply padding [PAD]
        if max_len is not None:
            pad_id = self.vocab.get("[PAD]", 0)
            #print(pad_id)
            if len(tokens) < max_len:
                # rigth padding
                tokens = tokens + [pad_id] * (max_len - len(tokens))
            else: tokens = tokens[:max_len] # truncate

        return tokens
    
    def sequences_to_ids(self, sequences: List[str], max_len : int = None) -> List[List[int]]:
        """
        Convert multiple sequences to token ids
        """

        return [self.tokenize(seq, max_len = max_len) for seq in sequences]
    
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

        
            
            






# testing
toy_sequences = [
    "AATCGTAGAATTGGGGGCCCCAATTC",
    "GTACCTAG",
    "AATGGCTA",
    "CGTAGCTA"
]

tokenizer = KmerTonenizer(k = 3, add_special_tokens = True)
tokenizer.build_vocab(toy_sequences)
#tokenizer.tokenize(toy_sequences[1], add_special_tokens = True)
max_len = max(len(seq) for seq in toy_sequences)
tok = tokenizer.sequences_to_ids(toy_sequences, max_len)
tok
len(tok)
len(tok[0])
len(tok[1])
tokenizer.ids_to_sequences(tok[1])
tokenizer.vocabulary_size

### Now apply for database
fasta_cds_list = [str(record.seq) for record in SeqIO.parse(open(coding), "fasta")]
fasta_ncds_list = [str(record.seq) for record in SeqIO.parse(open(non_coding), "fasta")]

cds_unique = list(set(fasta_cds_list))
ncds_unique = list(set(fasta_ncds_list))

cds_sorted = sorted(cds_unique, key = len)
ncds_sorted = sorted(ncds_unique, key = len)

cds_modicum = cds_sorted[25500:52500]
ncds_modicum = ncds_sorted[500:27500]
cds_modicum[1]

tokenizer.build_vocab(cds_modicum, ncds_modicum)
#max_len = max(len(seq) for seq in cds_modicum)
max_len = max(len(seq) for seq in ncds_modicum)
tokens_cds = tokenizer.sequences_to_ids(cds_modicum, max_len)
tokens_cds = np.array(tokens_cds)
tokens_cds.shape
memory_size = tokens_cds.nbytes / (1024**3)


tokens_ncds = tokenizer.sequences_to_ids(ncds_modicum, max_len)
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

### Buil 