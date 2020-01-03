# datasets
A dataset utils repository based on `tf.data`. **For tensorflow>=2.0 only!**

## Requirements

* python 3.6
* tensorflow>=2.0

## Installation

```bash
pip install nlp-datasets
```

## Usage

### seq2seq models

These models has an source sequence `x` and an target sequence `y`.

```python
from nlp_datasets import Seq2SeqDataset
from nlp_datasets import SpaceTokenizer
from nlp_datasets.utils import data_dir_utils as utils

files = [
    utils.get_data_file('iwslt15.tst2013.100.envi'),
    utils.get_data_file('iwslt15.tst2013.100.envi'),
]
x_tokenizer = SpaceTokenizer()
x_tokenizer.build_from_corpus([utils.get_data_file('iwslt15.tst2013.100.en')])
y_tokenizer = SpaceTokenizer()
y_tokenizer.build_from_corpus([utils.get_data_file('iwslt15.tst2013.100.vi')])
config = {
    'train_batch_size': 2,
    'predict_batch_size': 2,
    'eval_batch_size': 2,
    'buffer_size': 100
}
dataset = Seq2SeqDataset(x_tokenizer, y_tokenizer, config)

train_dataset = dataset.build_train_dataset(files)
print(next(iter(train_dataset)))
print('=' * 120)

eval_dataset = dataset.build_eval_dataset(files)
print(next(iter(eval_dataset)))
print('=' * 120)

predict_files = [utils.get_data_file('iwslt15.tst2013.100.envi')]
predict_dataset = dataset.build_predict_dataset(predict_files)
print(next(iter(predict_dataset)))
print('=' * 120)
```

Output:

```bash
(<tf.Tensor: id=328, shape=(2, 17), dtype=int64, numpy=
array([[628,  18,   3,  97,  96,   4,  10,  22,  52,   2,  18, 629,   0,
          0,   0,   0,   0],
       [628, 428, 112,  11,  26,  16,   8,   9, 134,  40, 429, 108,   3,
         33, 430,   2, 629]])>, <tf.Tensor: id=329, shape=(2, 19), dtype=int64, numpy=
array([[640,  54, 567,  16,  56,  83,   6,  15,  10,   9,   3,  54, 641,
          0,   0,   0,   0,   0,   0],
       [640, 181, 472, 291,  27,  47,  37, 112, 155, 188, 254,  45, 473,
         18,   1, 121, 145,   3, 641]])>)
========================================================================================================================
(<tf.Tensor: id=633, shape=(2, 21), dtype=int64, numpy=
array([[628,  42, 224,  30, 156,  59, 611, 612,   1,   5,  50,  81, 225,
         42, 613,  78, 208,   9, 614,   2, 629],
       [628,  91, 117, 448,   6,  27,  11,  26,  16,   8,  28, 449,   1,
          3, 200,   9, 450,   2, 629,   0,   0]])>, <tf.Tensor: id=634, shape=(2, 26), dtype=int64, numpy=
array([[640, 107,  12, 150, 312,  34, 101, 106, 325, 632, 317,   2,   5,
        633, 307,  35, 177, 107, 156, 175, 173,  85, 634,   3, 641,   0],
       [640, 225, 132,  21, 489, 490,  18,  27,  47,  37,  91,  22,  66,
         12, 491, 297,  70, 115,   1,   7, 204,   4, 298, 299,   3, 641]])>)
========================================================================================================================
tf.Tensor(
[[628  75   3   8  98   1   3  43   7  76   8   4 131  57   4 226   1   5
    3 227 132 228   9 229 230  18 231 232 233   2  18 629]
 [628 133   3   8  58 234   2 629   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0]], shape=(2, 32), dtype=int64)
========================================================================================================================
```

### sequence match models

These models has two sequences as input, `x` and `y`, and has an label `z`.

```python
from nlp_datasets import SeqMatchDataset
from nlp_datasets import SpaceTokenizer
from nlp_datasets.utils import data_dir_utils as utils

files = [
    utils.get_data_file('dssm.query.doc.label.txt'),
    utils.get_data_file('dssm.query.doc.label.txt'),
]
x_tokenizer = SpaceTokenizer()
x_tokenizer.build_from_vocab(utils.get_data_file('dssm.vocab.txt'))
y_tokenizer = SpaceTokenizer()
y_tokenizer.build_from_vocab(utils.get_data_file('dssm.vocab.txt'))

config = {
    'train_batch_size': 2,
    'eval_batch_size': 2,
    'predict_batch_size': 2,
    'buffer_size': 100,
}
dataset = SeqMatchDataset(x_tokenizer, y_tokenizer, config)

train_dataset = dataset.build_train_dataset(files)
print(next(iter(train_dataset)))
print('=' * 120)

eval_dataset = dataset.build_eval_dataset(files)
print(next(iter(eval_dataset)))
print('=' * 120)

predict_files = [utils.get_data_file('dssm.query.doc.label.txt')]
predict_dataset = dataset.build_predict_dataset(predict_files)
print(next(iter(predict_dataset)))
print('=' * 120)
```

Output:

```bash
(<tf.Tensor: id=514, shape=(2, 5), dtype=int64, numpy=
array([[10,  1,  3,  4, 11],
       [10,  1,  3,  4, 11]])>, <tf.Tensor: id=515, shape=(2, 11), dtype=int64, numpy=
array([[10,  0,  1,  2,  7,  5,  8,  6,  3,  9, 11],
       [10,  0,  1,  2,  7,  5,  8,  6,  3,  9, 11]])>, <tf.Tensor: id=516, shape=(2,), dtype=int64, numpy=array([1, 0])>)
========================================================================================================================
(<tf.Tensor: id=920, shape=(2, 5), dtype=int64, numpy=
array([[10,  1,  3,  4, 11],
       [10,  1,  3,  4, 11]])>, <tf.Tensor: id=921, shape=(2, 11), dtype=int64, numpy=
array([[10,  0,  1,  2,  7,  5,  8,  6,  3,  9, 11],
       [10,  0,  1,  2,  7,  5,  8,  6,  3,  9, 11]])>, <tf.Tensor: id=922, shape=(2,), dtype=int64, numpy=array([0, 1])>)
========================================================================================================================
(<tf.Tensor: id=1206, shape=(2, 5), dtype=int64, numpy=
array([[10,  1,  3,  4, 11],
       [10,  1,  3,  4, 11]])>, <tf.Tensor: id=1207, shape=(2, 11), dtype=int64, numpy=
array([[10,  0,  1,  2,  7,  5,  8,  6,  3,  9, 11],
       [10,  0,  1,  2,  7,  5,  8,  6,  3,  9, 11]])>)
========================================================================================================================
```

### sequence classify model

These models has a input sequence `x`, and a output label `y`.

```python
from nlp_datasets import SeqClassifyDataset
from nlp_datasets import SpaceTokenizer
from nlp_datasets.utils import data_dir_utils as utils

files = [
    utils.get_data_file('classify.seq.label.txt')
]
x_tokenizer = SpaceTokenizer()
x_tokenizer.build_from_corpus([utils.get_data_file('classify.seq.txt')])

config = {
    'train_batch_size': 2,
    'eval_batch_size': 2,
    'predict_batch_size': 2,
    'buffer_size': 100
}
dataset = SeqClassifyDataset(x_tokenizer, config)

train_dataset = dataset.build_train_dataset(files)
print(next(iter(train_dataset)))
print('=' * 120)

eval_dataset = dataset.build_eval_dataset(files)
print(next(iter(eval_dataset)))
print('=' * 120)

predict_files = [utils.get_data_file('classify.seq.txt')]
predict_dataset = dataset.build_predict_dataset(predict_files)
print(next(iter(predict_dataset)))
print('=' * 120)
```

Output:

```bash
(<tf.Tensor: id=349, shape=(2, 7), dtype=int64, numpy=
array([[7, 1, 4, 5, 6, 2, 8],
       [7, 1, 3, 2, 8, 0, 0]])>, <tf.Tensor: id=350, shape=(2,), dtype=int64, numpy=array([0, 1])>)
========================================================================================================================
(<tf.Tensor: id=601, shape=(2, 7), dtype=int64, numpy=
array([[7, 1, 3, 2, 8, 0, 0],
       [7, 1, 4, 5, 6, 2, 8]])>, <tf.Tensor: id=602, shape=(2,), dtype=int64, numpy=array([1, 0])>)
========================================================================================================================
tf.Tensor(
[[7 1 3 2 8 0 0]
 [7 1 4 5 6 2 8]], shape=(2, 7), dtype=int64)
========================================================================================================================
```
