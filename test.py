import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torch.utils.data import DataLoader

import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Optional imports - only import if available
try:
    import spacy
except ImportError:
    spacy = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


################################################################################
# Some convenience helper functions used throughout the notebook


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


################################################################################
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    the encoder maps an input sequence of symbol representations 
    (x_1, ..., x_n) to a sequence of continuous representations, z=
    (z_1, ..., z_n).
    Encoder-Decoder architecture changes input sequences into vectors,
    vice-versa.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask) 
        # encode and decode
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask) 
        # sequences to vectors, vector lengths are the same
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask) 
        # vectors to sequences


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        """
        linear transformation, x * W^T + b, W as weight and b as bias, d_model is the size 
        of the input tensor, vocab is the size of the output tensor. for an input tensor like 
        (batch_size, seq_length, input_size), tensor[-1] would be d_model.
        """
        super(Generator, self).__init__() 
        #make initialisation global
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x):
        """
        Here we take the log of the partition function, log(exp(x_i) / sum(exp(x_j) for j in
        range(len(x). This avoids data overflow.
        """
        return F.log_softmax(self.proj(x), dim=-1)
    

################################################################################
def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]) 
    # create N copies of module as a ModuleList

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) 
        # make "N" deep copies of layer
        self.norm = nn.LayerNorm(layer.size) 
        # normalise every single layer
        
    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask) 
            # define layers
        return self.norm(x) 
        # normalise
    

################################################################################
class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See arXiv:1607.06450). The goal is to calculate mean and 
    stddev for every single sample with their own characteristic dimension.
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        """
        nn.Parameter save parameters as part of nn.Module, it is included in gradiant 
        calculation so it can be traced and updated during training sessions
        """
        self.a_2 = nn.Parameter(torch.ones(features)) 
        # create tensors with all elements equal to 1 with size "features"
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """Specific function to make convergence faster."""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * ((x - mean) / (std + self.eps)) + self.b_2 
        # first term and last term made Affine Transformation, the term in the middle is normalisation term.
    

################################################################################
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout) 
        # randomly drop some layers to avoid producing same results, fraction == dropout

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))
    

################################################################################    
class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) 
        # 2 copies of the sublayers defined as dropout added to tensor 'size'.
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        """x is used to control the input tensor with Causal Self Attention with mask,
        x in self_attn(x, x, x, mask) means query, key and value."""
        return self.sublayer[1](x, self.feed_forward) 
    

################################################################################
class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """memory: output from original sequences. src_mask and tgt_mask: the masks for source
        and target sequences."""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask) 
            # mark the layer as x
        return self.norm(x)
    

################################################################################
class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below).
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Forward the combined layer from source and target with mask. 
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) 
        # mark masked sublayer as x
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # used x calculated before to find correlation between two layers
        return self.sublayer[2](x, self.feed_forward)
        

################################################################################
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # size of the mask, with size of (batch_size, sequence_length, sequence_length)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.bool)
    # set a matrix with 1 on the triangle below the diagonal and 0 on the upper triangle.
    return subsequent_mask == 0
    # change the matrix into tensor and shield the element if the element == 0.


################################################################################
"""
Below the attention mask shows the position each tgt word (row) is allowed to look at (column). Words are 
blocked for attending to future words during training.
"""
def example_mask():
    LS_data = pd.concat(# combining dataframes into one
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    # flatten enables further processing like calculating loss of data
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )
    #making a dataframe of size 20*20
    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_mask)


################################################################################
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) 
    # query usually equals to (batch_size, sequence_length, embedding_dim), here returns embedding_dim
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # dot product of query and key, devided by math.sqrt(d_k) to avoid overflow that fails softmax function
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # for the same reason, replace mask with very small number to avoid uneffective softmax functions
    p_attn = F.softmax(scores, dim=-1) 
    # apply softmax function along the last dimension
    if dropout is not None:
        p_attn = dropout(p_attn) 
        # apply dropout to weight if there is one
    return torch.matmul(p_attn, value), p_attn
    # return the product of weight and tensor, which is the attension output, and the weight


################################################################################
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # the d_model is splitted into h segments, each of size d_k.
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        """4 identical linear sublayers, linear transform with queries, keys and values, 
        then combine all and transform on the last layer."""
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None # save the attention weight
        self.dropout = nn.Dropout(p=dropout) # just a dropout
        
    def forward(self, query, key, value, mask=None):
        "extract Q, K, V, apply attention and feed forward to the last layer"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # insert a dimension after first dimension
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        """
        zip() created a dictionary for linear layer corresponding to (query, key, value);
        l, x means layers and tensors;
        l(x) means transfer x into l, linear layer;
        view() gives (nbatch, num of heads, seq lengths, dim per head);
        transpose() exchange two dimensions in position 1 and 2.
        """
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout) # attention output & weight
        
        # 3) "Concat" using a view and apply a final linear. 
        """
        contiguous creates a new tensor in which variables are saved continuously.
        self.h * self.d_k combined all outputs to a new tensor
        """
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value # release the memory
        return self.linears[-1](x) # send x to the last layer
    

################################################################################
class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        """use higher dimensional intermediate layer, d_ff, to make linear transform more 
        variable"""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        F.relu() applies activation function to the linear layer w_1(x);
        the final result for w_1 is transferred to w_2
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

################################################################################
class Embeddings(nn.Module):
    """project discrete inputs into continuous vectors"""
    def __init__(self, d_model, vocab):
        """
        d_model defines length of vectors for each discrete input;
        vocab defines how much discrete values the inputs may take 
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """times embedding layer with sqrt(self.d_model) to avoid overflow"""
        return self.lut(x) * math.sqrt(self.d_model)
    

################################################################################
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        """
        to make tensor dimensionally matched, unsqueeze() make 1-d tensor torch.arange(0, max_len)
        a 'transpose tensor', which is 2-d, and elements are all the integers in range(0, max_len-1).
        """
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        """
        tensor takes step == 2 to generate cosine and sine waves using 2i and 2i+1;
        exp(-(math.log(10000.0) / d_model))) gives waves with different frequencies;
        low frequency means longer wavelengths, having more words analysed together when the 
        wave is in positive values, so it focused more on long-distance correlations;
        high frequency means shorter wavelengths, having less words but higher frequency for 
        analysis when the wave is in positive values, so it focused more on short-distance 
        correlations;
        """
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        """
        buffer is saved as state_dict, it is part of the model and it will not be modified or 
        calculated;
        'pe' is the name and pe is the tensor saved;
        """
        
    def forward(self, x):
        """
        self.pe[:, :x.size(1)] finds corresponding position encoding based on given sequence
        length;
        Variable( , requires_grad=False) cancels gradient calculation for position encoding,
        prevents reversed propagation.
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
    

################################################################################
def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20)) # positional encoding tensor (torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7] # create dataframe for each dimension
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )

show_example(example_positional)


################################################################################
# Model construction helper function
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                           c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


################################################################################
# Example usage
if __name__ == "__main__":
    # Create a small model for testing
    tmp_model = make_model(10, 10, 2)
    print("Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in tmp_model.parameters())}")
