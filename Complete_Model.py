#%%
with open('transformer.py', 'r') as f:
    code = f.read()
exec(code)
"""
import function failed on my system so transformer.py was run as part of this file here.
"""


#%%
################################################################################
"""
Full Model
"""

def make_model(src_vocab, tgt_vocab, N=6, 
                d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    # to have a more completed global understanding and faster processing
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    """
    introduce non-linear activation function for more complicated expression, better
    ability in extracting features in detail and fix the shortage of MHA, which cares
    about global features only.
    """
    position = PositionalEncoding(d_model, dropout)
    """
    for transformer, existance of self attention means processing multiple sequences 
    at the same time. position encoding makes the relative position of the sequences 
    more detectable and avoids requirements for re-train when applying different 
    lengths to it.
    """
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    """
    self-attention focus on one single sequence, used in encoder only. encoder-decoder 
    attention focus on relationsip between every single word between target sequence and 
    source sequence. similarity is obtained by weight of query to the key, illustrated by
    direction of their unit vectors.
    generator uses linear transformation to change its dimension from d_model to target vocab
    table. applying softmax could change this into probability distribution.
    """
    
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    """
    nn.init.xavier_uniform_(p) means Glorot uniform distribution in neural network system. 
    it takes random sample value in limits with range +-(sqrt(6 / (fan_in + fan_out))). fan_in
    means neural number from the previous layer and fan_out means neural number in the current 
    layer. Except for ReLU, where the activation function is not symmetric, the Glorot uniform 
    distribution prevents gradient vanish or overflow and increases the speed of convergence.
    """
    return model


################################################################################
"""
Inference
"""

"""
Here we make a forward step to generate a prediction of the model. We try to use our transformer
to memorize the input. As you will see the output is randomly generated due to the fact that the
model is not trained yet. In the next section we will build the training function and try to 
train our model to memorize the numbers from 1 to 10.
"""
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval() # evaluation mode shall disable the mechanism like dropout for stable outputs
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9): # generate 9 tokens
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        ) # generate the token with decoder, causal effect by subsequent_mask
        prob = test_model.generator(out[:, -1]) # probability distribution of the next token
        _, next_word = torch.max(prob, dim=1) # find most likely token
        next_word = next_word.data[0] # set value of token as next_word
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        ) # add token into "ys"

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)


################################################################################
"""
Training
"""

"""
We stop for a quick interlude to introduce some of the tools needed to train a 
standard encoder decoder model. First we define a batch object that holds the src 
and target sentences for training, as well as constructing the masks.
"""

"""
Batches and Masking
"""
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2): 
        # 2 = <blank>, index for padding tokens
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) 
        """
        creating mask with form [batch_size, 1, seq_len], insert a dimension before
        2nd last one.
        """
        if tgt is not None:
            self.tgt = tgt[:, :-1] # save copy of target[:-1] as input of decoder
            self.tgt_y = tgt[:, 1:] # save copy of target[1:] to calculate the loss
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
            """calculate actual quantity of tokens in the current batch, exclude 
            the influence from padding tokens to make normalisation or giving weight 
            easier in the later stages of training, like benchmarks or calculating
            the loss.
            """

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    

"""
Training loop
"""
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # Total # of examples used
    tokens: int = 0  # Total # of tokens processed

def run_epoch( 
    data_iter, # load training data by batches
    model, 
    loss_compute, 
    optimizer, # updating the parameters of the model
    scheduler, # adjust the rate of learning
    mode="train",
    accum_iter=1,
    train_state=TrainState(), # record information during the training
):
    """Train a single epoch"""
    """
    one epoch means a complete traversal across the whole dataset for once and 
    model will process every batch of data. the model will be optimized step by
    step during forward/backward propagating calculating loss and updating 
    parameters.
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    "initialise the parameters"
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        ) # propagating forward
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0] # number of samples
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0: # update parameter with accumulated steps
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) # clear gradient
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step() # update learning rate

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            # print every 40 steps
            lr = optimizer.param_groups[0]["lr"] 
            """
            separate parameters into different groups and set different parameters
            for different groups. this is useful when you only want model to be 
            partially updated or fine tuning, where only some of the parameters 
            needs to be effective for specific jobs.
            """
            elapsed = time.time() - start # token/s
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                ) # format placeholder for different dtypes
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


################################################################################
"""
Optimizer
"""

"""
optimizer uses gradient of loss function to update the parameters of the model.
here we apply Adam optimizer(Adaptive Moment Estimation). It has both advantages 
from momentum, which accelerates convergence and reduces oscillation by maintaining
average change of gradient for different parameters, and self-adaptive learning 
rate by maintaining average change of gradient squared to deal with dradient 
overflow or vanish.
"""
def rate(step, model_size, factor, warmup): # learning rate
    """
    We have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1 # avoid overflow when calculating 0**(-0.5)
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
"""
model_size ** (-0.5): a scaling factor for adjusting the relationship between the
learning rate and the size of the model.
step ** (-0.5): as the step increases, the learning rate decreases.
step * warmup ** (-1.5): an equation to calculate the learning rate. warmup means 
the total step for warmup.
"""

def example_learning_schedule():
    """
    illustrates change of learning rate under different model size and warmup steps.
    """
    opts = [
        [512, 1, 4000],  # example 1, each follows [model_size, factor, warmup]
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1) # a simple model, to get optimizer only
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        """
        large learning rate means faster convergence, meanwhile less stable updates.
        large beta() values means more stable gradient change, calculated by: v_t 
        = beta * v_{t-1} + (1 - beta) * x_t
        eps takes very small value to avoid overflow
        """
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        ) 
        # *example includes lr or steps, decay step means lr decays as step increases
        tmp = []
        # Take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)
    
    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()
    """
    override the default limit of the number of rows in a dataset that Altair 
    processes for visualization (5000).
    """
    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2] # columns in the dataframe
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )

example_learning_schedule()


################################################################################
"""
Regularization
"""

"""
sometimes the data was learned by too much detail rather than the general rule. 
this makes the model unable to be trained with other datasets.
regularization prevents overfitting and make model more simple and more various
to have better performance under different datasets. label smoothing is one of
the methods for regularization.
"""

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    """
    here we used label smoothing of value ϵ_ls = 0.1, which means all categories
    share 0.1 of total probabilities and the other 0.9 for the categories for the
    sample.
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        # padding index will be ignored. smoothing is the level of smoothing.
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        """
        KLDiv is calculated by KL(P || Q) = Σ P(x) * log(P(x) / Q(x)). P is actual
        probability distribution and Q is expected probability distribution. 
        reduction='mean' means mean reduction of the model, reduction='sum' means 
        sum of reduction for the modeland reduction='none' gives reduction by every
        single element.
        """
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        """
        normally smoothing factor takes value from 0.0 to 0.1 and it controls level 
        of noises learned by the model. we take larger smothing factor for models
        that is easier to be overfitted, this includes small datasets, complicated
        model and completing complicated missions.
        """
        self.size = size
        self.true_dist = None # true distribution after smoothing

    def forward(self, x, target): 
        # propagate forward, gives loss between actual data and true_dist
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        # probability given to non-true categories
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # unsqueeze(1) makes data to be in same form as true_dist
        true_dist[:, self.padding_idx] = 0
        # padding index should not be counted as part of the loss calculation
        mask = torch.nonzero(target.data == self.padding_idx)
        """
        self.padding_idx: uses padding tokens to fill the shorter sentences to 
        the same length as the longest one.
        mask gives the position of padding token in the target, target.data == 
        self.padding_idx compares elements in the target with self.padding_idx
        and return a bool tensor with same form as the target.
        """
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            """
            0 is position to fill (row), mask.squeeze() is the index list for the 
            rows, and 0.0 is the value for padding. the filling executes in the 
            original tensor.
            """ 
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
    """
    this gives loss between expected value by model and actual loss.clone().detach() 
    avoids changing the true_dist value through backward propagation.
    """
    
# Example of label smoothing.

def example_label_smoothing(): # take log probability of expected input
    crit = LabelSmoothing(5, 0, 0.4)
    # 5 categories, no padding index, 0.4 as smoothing factor
    predict = torch.FloatTensor( # expected output
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    """
    x=predict.log() since expected input is a log of probabilities
    torch.LongTensor saves data as a 64 digit integer
    [2, 1, 0, 3, 3] is the list for 5 categories listed before.
    """
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    # crit.true_dist is the distribution after smoothing.
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )

show_example(example_label_smoothing)

"""
Label smoothing actually starts to penalize the model if it gets very confident 
about a given choice.
"""

def loss(x, crit):
    d = x + 3 * 1 # probability for regularisation
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    # [0, x / d, 1 / d, 1 / d, 1 / d] is predicted probabilities for the categories
    return crit(predict.log(), torch.LongTensor([1])).data
    # torch.LongTensor([1]) suggests 2nd category is true


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )

show_example(penalization_visualization)


################################################################################
"""First Example"""

"""
We can begin by trying out a simple copy-task. Given a random set of input symbols 
from a small vocabulary, the goal is to generate back those same symbols.
"""

"""Synthetic Data"""

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    """
    V: list of volcabularies 
    """
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        """
        size=(batch_size, 10) means generating a tensor with size (batch_size, 10).
        number of tensors is a random integer from 1 to V.
        """
        data[:, 0] = 1 # adding a start sign for each sequence
        src = data.requires_grad_(False).clone().detach()
        # make sure thi sonly provides data, no training needed here
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)
        # yield: this generates one result per time to save memory.

"""Loss Computation"""

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator # prediction for outputs
        self.criterion = criterion 
        # calculate loss between expected and actual outputs

    def __call__(self, x, y, norm):
        """
        x: inputs for model
        y: actual outputs
        norm: normalisation factors
        """
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm # normalisation
        )
        """
        view() requires a continuous storage so contiguous() is applied. 
        (-1, x.size(-1)) changes tensor into 2-dimension with first dimension as -1
        and second dimension as x.size(-1).
        """
        return sloss.data * norm, sloss 
    """
    original data and normalised data for calculating gradiant of backward propagation
    """


################################################################################
"""Greedy Decoding"""

"""
the iterator is given by directly adding the word with greatest probability to the
end of queue as output until ending token or maximum length is reached.
it is quick and easier to be maintained as only one sequence is processed. this
also means invariance and regardless for overall sequence.
Beam Search or sampling could be possible improvements to this with better variance
of reaults.
"""

"""This code predicts a translation using greedy decoding for simplicity."""

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    # store the generated output sequence
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        """
        subsequent_mask(ys.size(1)).type_as(src.data) creates a self-recurrent
        mask to prevent casual conflicts
        """
        prob = model.generator(out[:, -1])
        """
        probability distribution of next word, by generator from last time step,
        usually a linear layer with softmax
        """
        _, next_word = torch.max(prob, dim=1)
        """
        return the maximum value in each line.
        _: exact probability
        next_word: index for the value
        """
        next_word = next_word.data[0] 
        # get original data, only needed for older version of PyTorch
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        """
        torch.cat: combines multiple tensors into a larger one along a specific
        axis, dim = 1 in this case.
        torch.zeros(1, 1).type_as(src.data).fill_(next_word): this means construct
        a tensor with size (1,1) and make it has same datatype as src.data and 
        finally (next_word, next_word)
        """
    return ys

# Train the simple copy task.

def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR( # adjust lr
        optimizer=optimizer,
        lr_lambda=lambda step: rate( # each step has its own lr
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20), # 20 is maximum length
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5), # generate next batch
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(), # padding optimizer, no update to the data
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


execute_example(example_simple_model)