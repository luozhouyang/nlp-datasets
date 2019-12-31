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
