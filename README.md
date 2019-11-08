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

```python
from nlp_datasets import XYSameFileDataset
from nlp_datasets import SpaceTokenizer

tokenizer = SpaceTokenizer()
dataset = XYSameFileDataset(x_tokenizer=tokenizer, y_tokenizer=tokenizer, config=None)
train_files = [...]
train_dataset = dataset.build_train_dataset(train_files=train_files)
```