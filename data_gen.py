# /// script
# dependencies = [
#   "numpy",
#   "torch",
# ]
# ///

import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader


### ===== Config =====
VOCAB_SIZE = 65
INPUT_SEQ_LEN = 64
BATCH_SIZE = 256
NUM_KV_PAIRS = 16
NUM_TRAIN_EXAMPLES = 100
NUM_TEST_EXAMPLES = 30
TRAIN_POWER_A = 0.01
TEST_POWER_A = 0.01
RANDOM_NON_QUERIES = True
SEED = 0
SAVE_PATH = "data.pt"

### ================

### ===== Zoology Code =====

# copied as is from zoology/data/utils.py
@dataclass
class SyntheticData:
    """Simple dataclass which specifies the format that should be returned by
    the synthetic data generators.

    All tensors (train_inputs, train_labels, test_inputs, test_labels) should be
    have two axes and share the same second dimension length.

    Args:
        train_inputs (torch.Tensor): Training inputs of shape (num_train_examples, input_seq_len)
        train_labels (torch.Tensor): Training labels of shape (num_train_examples, input_seq_len)
        test_inputs (torch.Tensor): Test inputs of shape (num_test_examples, input_seq_len)
        test_labels (torch.Tensor): Test labels of shape (num_test_examples, input_seq_len)
    """

    train_inputs: torch.Tensor
    train_labels: torch.Tensor
    test_inputs: torch.Tensor
    test_labels: torch.Tensor

    def check_shapes(
        self,
        num_train_examples: int,
        num_test_examples: int,
        input_seq_len: int,
    ):
        """Check that the shapes are correct
        this is useful to catch bugs in the data generation code because
        downstream errors due to incorrectly shaped can be tricky to debug.
        """
        if self.train_inputs.shape != (num_train_examples, input_seq_len):
            raise ValueError(
                f"train_inputs shape is {self.train_inputs.shape} but should be {(num_train_examples, input_seq_len)}"
            )

        if self.train_labels.shape != (num_train_examples, input_seq_len):
            raise ValueError(
                f"train_labels shape is {self.train_labels.shape} but should be {(num_train_examples, input_seq_len)}"
            )

        if self.test_inputs.shape != (num_test_examples, input_seq_len):
            raise ValueError(
                f"test_inputs shape is {self.test_inputs.shape} but should be {(num_test_examples, input_seq_len)}"
            )

        if self.test_labels.shape != (num_test_examples, input_seq_len):
            raise ValueError(
                f"test_labels shape is {self.test_labels.shape} but should be {(num_test_examples, input_seq_len)}"
            )

# copied as is from zoology/data/recent_associative_recall.py
def multiquery_ar(
    vocab_size: int=8_192,
    num_train_examples: int=100_000,
    num_test_examples: int=3_000,
    input_seq_len: int=64,
    num_kv_pairs: int=4,
    train_power_a: float=0.01,
    test_power_a: float=0.01,
    random_non_queries: bool=True,
    seed: int=0,
) -> SyntheticData:
    """
    Generates synthetic data for the multi-query associative recall task as described in
    Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

    Example: 
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)` 
        will generate input and label sequences of the form: 
                
                Key   Val  Key  Val            Query                         Query
        Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0 
        Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

        The -100 labels are ignored by the loss function and metrics.
    
    We include one important note on the power law distribution. In real language data, 
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again 
    drops the further away from the orginal mention we are. In our synthetic, we can 
    control this with the power law parameters `train_power_a` and `test_power_a`. 
    Setting these to 1.0 will result in a uniform distribution. You can visualize the
    distribution with the following code:
    ```
    space = 100
    power_a = 0.01  
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    plt.plot(p)
    ```

    Args:
        vocab_size (int): The size of the vocabulary. As discussed in the Zoology 
            paper, large vocabulary sizes (>1k) can be important for highlighting 
            differences between model architectures. Defaults to 8_192.
        num_train_examples (int): The number of training examples to generate. Defaults 
            to 100_000.
        num_test_examples (int): The number of test examples to generate. Defaults to 
            3_000.
        input_seq_len (int): The length of the input sequence. Defaults to 64. In 
            In Figure 2 of the Zoology paper, we vary the input sequence length from 
            64 to 512 and the number of key-value pairs from 4 to 64.
        seed (int): The seed for the random number generator.
        num_kv_pairs (int): The number of key-value pairs.
        train_power_a (float, optional): The power for the power law distribution for 
            training data. Defaults to 0.01.
        test_power_a (float, optional): The power for the power law distribution for 
            test data. Defaults to 0.01.
        random_non_queries (bool, optional): If True, replace all the 0's (as in the 
            example above) with random values in the input. Defaults to True.

    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test 
            inputs and labels.

    Raises:
        Warning: If potential data leakage is detected between the train and test sets.
    """

    train_inputs, train_labels = _mqar(
        vocab_size=vocab_size,
        num_examples=num_train_examples,
        input_seq_len=input_seq_len,
        seed=seed,
        power_a=train_power_a,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=random_non_queries
    )

    test_inputs, test_labels = _mqar(
        vocab_size=vocab_size,
        num_examples=num_test_examples,
        input_seq_len=input_seq_len,
        seed=seed + 10,  # different seed for test set
        power_a=test_power_a,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=random_non_queries
    )

    data = SyntheticData(
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
    )

    # check for data leakage:
    train_set = set([" ".join(map(str, x)) for x in data.train_inputs.tolist()])
    test_set = set([" ".join(map(str, x)) for x in data.test_inputs.tolist()])
    frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
    if frac_test_in_train > 0.001:
        print(
            "WARNING: Potential data leakage detected. " 
            f"{frac_test_in_train: 0.2f} of test examples are in the train set."
        )
    return data


def _mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,
    random_non_queries: bool=True
):
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=True, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=True, size=num_kv_pairs)
    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(
        queries, (gaps * 2),
        values=np.apply_along_axis(np.random.choice, 1, keys, replace=True, size=keys[0].shape),
        axis=1
    )
    examples = np.concatenate([kvs, queries], axis=1)
    inputs = torch.tensor(examples[:, :-1])

    if random_non_queries:
        inputs[inputs == 0] = torch.tensor(
            np.random.choice(value_choices, replace=True, size=inputs[inputs == 0].shape))

    def process_sequence(seq):
        out = np.full((seq.shape[0],), -100, dtype=np.int64)
        state = {}
        curr_key = None
        for i in range(seq.shape[0]):
            if seq[i] in key_choices:
                curr_key = seq[i]
                if curr_key in state:
                    out[i] = state[curr_key]
            elif curr_key is not None:
                state[curr_key] = seq[i]
                curr_key = None
        return out
    labels = torch.tensor(
        np.apply_along_axis(process_sequence, axis=1, arr=inputs))
    # labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    # labels = np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)
    # inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    # replace all the 0 with random values
    return inputs, labels

# ========================
    

# generate data
data = multiquery_ar(
    vocab_size=VOCAB_SIZE,
    num_train_examples=NUM_TRAIN_EXAMPLES,
    num_test_examples=NUM_TEST_EXAMPLES,
    input_seq_len=INPUT_SEQ_LEN,
    num_kv_pairs=NUM_KV_PAIRS,
    train_power_a=TRAIN_POWER_A,
    test_power_a=TEST_POWER_A,
    random_non_queries=RANDOM_NON_QUERIES,
    seed=SEED,
)

train_dl = DataLoader(
    TensorDataset(data.train_inputs, data.train_labels),
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
)
test_dl = DataLoader(
    TensorDataset(data.test_inputs, data.test_labels),
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
)

print(f"\nStandalone Train DataLoader: {len(train_dl)} batches")
print(f"Standalone Test DataLoader: {len(test_dl)} batches")

# Get a sample batch
for batch_inputs, batch_labels in train_dl:
    print(f"\nStandalone Sample batch shape:")
    print(f"  Inputs: {batch_inputs.shape}")
    print(f"  Labels: {batch_labels.shape}")
    print(f"\nFirst example from batch:")
    print(f"  Input: {batch_inputs[0]}")
    print(f"  Label: {batch_labels[0]}")
    break

# save the dataset to a file
torch.save(data, SAVE_PATH)
print(f"Dataset saved to {SAVE_PATH}")

# HOW TO: load the dataset from the file
# from data_gen import SyntheticData
data = torch.load(SAVE_PATH, weights_only=False)
print(f"Dataset loaded from {SAVE_PATH}")

train_dl = DataLoader(
    TensorDataset(data.train_inputs, data.train_labels),
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
)
test_dl = DataLoader(
    TensorDataset(data.test_inputs, data.test_labels),
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
)

# Get a sample batch
for batch_inputs, batch_labels in train_dl:
    print(f"\nStandalone Sample batch shape:")
    print(f"  Inputs: {batch_inputs.shape}")
    print(f"  Labels: {batch_labels.shape}")
    print(f"\nFirst example from batch:")
    print(f"  Input: {batch_inputs[0]}")
    print(f"  Label: {batch_labels[0]}")
    break
