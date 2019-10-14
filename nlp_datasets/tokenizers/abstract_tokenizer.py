import abc
import logging
import os

import tensorflow as tf
from collections import Counter


class AbstractTokenizer(abc.ABC):
    """Tokenizer for language. The first line of vocab file is always `0   <UNK>`."""

    def __init__(self, config=None):
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config

        self._special_tokens = [
            self.config['unk_token'],
            self.config['sos_token'],
            self.config['eos_token'],
            self.config['cls_token'],
            self.config['sep_token'],
            self.config['mask_token']
        ]

        self.counter = Counter()
        self._id2token_dict = None
        self._token2id_dict = None
        self._id2token_table = None
        self._token2id_table = None

    def _process_line(self, line, token_filters=None):
        """Process line by line in file.

        Args:
            line: A python str, line of file
            token_filters: An iterable of TokenFilter
        """
        raise NotImplementedError()

    @property
    def vocab_size(self):
        return len(self._token2id_dict)

    @property
    def unk_id(self):
        return 0

    @property
    def unk_token(self):
        return self.config.get('unk_token', '<UNK>')

    @property
    def sos_id(self):
        return self._token2id_dict.get(self.config['sos_token'], None)

    @property
    def sos_token(self):
        if not self.config['add_sos']:
            return None
        return self.config.get('sos_token', '<SOS>')

    @property
    def eos_id(self):
        return self._token2id_dict.get(self.config['eos_token'], None)

    @property
    def eos_token(self):
        if not self.config['add_eos']:
            return None
        return self.config.get('eos_token', '<EOS>')

    @property
    def cls_id(self):
        return self._token2id_dict.get(self.config['cls_token'], None)

    @property
    def cls_token(self):
        if not self.config['add_cls']:
            return None
        return self.config.get('cls_token', '[CLS]')

    @property
    def sep_id(self):
        return self._token2id_dict.get(self.config['sep_token'], None)

    @property
    def sep_token(self):
        if not self.config['add_sep']:
            return None
        return self.config.get('sep_token', '[SEP]')

    @property
    def mask_id(self):
        return self._token2id_dict.get(self.config['mask_token'], None)

    @property
    def mask_token(self):
        if not self.config['add_mask']:
            return None
        return self.config.get('mask_token', '[MASK]')

    @property
    def token2id_dict(self):
        return self._token2id_dict

    @property
    def id2token_dict(self):
        return self._id2token_dict

    def encode(self, tokens):
        """Encode string tokens to ids."""
        return self._token2id_table.lookup(tokens)

    def decode(self, ids):
        """Decode ids to string tokens."""
        return self._id2token_table.lookup(ids)

    def _reset(self):
        self.counter.clear()
        self._id2token_dict = dict()
        self._token2id_dict = dict()
        logging.info('Tokenizer reset.')

    def build_from_corpus(self, corpus_files, max_vocab_size=None, token_filters=None):
        """Build lookup table and vocab dict from corpus files.

        Args:
            corpus_files: An iterable of files
            max_vocab_size: Max size of vocab, not including special tokens like unk|sos|eos and so on.
            token_filters: An iterable of TokenFilter instances
        """
        self._reset()

        for f in corpus_files:
            if not os.path.exists(f):
                logging.warning('File %s does not exist.' % f)
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip('\n')
                    if not line:
                        continue
                    self._process_line(line, token_filters)

        self._token2id_dict = {self.config['unk_token']: 0}

        # select most common vocabs
        for k, v in self.counter.most_common(max_vocab_size):
            self._token2id_dict[k] = len(self._token2id_dict)

        logging.info("Vocab size including unk is: %d" % len(self._token2id_dict))
        self._add_special_tokens(self._token2id_dict)
        logging.info("Vocab size including special tokens is: %d" % len(self._token2id_dict))

        for k, v in self._token2id_dict.items():
            self._id2token_dict[v] = k

        self._init_lookup_tables(self._token2id_dict, self._id2token_dict)
        logging.info("Build tokenizer from corpus files finished.")

    def build_from_vocab(self, vocab_file):
        """Build lookup table from vocab file."""
        self._reset()

        with open(vocab_file, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.strip('\n').strip()
                if not line:
                    continue
                if line in self._special_tokens:
                    continue
                self._token2id_dict[line] = len(self._token2id_dict)

        logging.info("Vocab size including unk is: %d" % len(self._token2id_dict))
        self._add_special_tokens(self._token2id_dict)
        logging.info("Vocab size including special tokens is: %d" % len(self._token2id_dict))

        for k, v in self._token2id_dict.items():
            self._id2token_dict[v] = k

        # init lookup tables
        self._init_lookup_tables(self._token2id_dict, self._id2token_dict)
        logging.info("Build tokenizer from vocab file finished.")

    def _add_special_tokens(self, token2id):
        if self.config['add_sos']:
            token2id[self.config['sos_token']] = len(token2id)
        if self.config['add_eos']:
            token2id[self.config['eos_token']] = len(token2id)
        if self.config['add_cls']:
            token2id[self.config['cls_token']] = len(token2id)
        if self.config['add_sep']:
            token2id[self.config['sep_token']] = len(token2id)
        if self.config['add_mask']:
            token2id[self.config['mask_token']] = len(token2id)

    def _init_lookup_tables(self, token2id_dict, id2token_dict):
        token2id_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=list(token2id_dict.keys()),
            values=list(token2id_dict.values()),
            key_dtype=tf.dtypes.string,
            value_dtype=tf.dtypes.int64)
        self._token2id_table = tf.lookup.StaticHashTable(
            initializer=token2id_initializer,
            default_value=0,  # unk id
            name='token2id_lookup_table')

        id2token_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=list(id2token_dict.keys()),
            values=list(id2token_dict.values()),
            key_dtype=tf.dtypes.int64,
            value_dtype=tf.dtypes.string)
        self._id2token_table = tf.lookup.StaticHashTable(
            initializer=id2token_initializer,
            default_value=self.config.get('unk_token', '<UNK>'),
            name='id2token_lookup_table')

    def save_to_vocab(self, output_file):
        with open(output_file, mode='wt', encoding='utf8') as fout:
            for k, v in self._id2token_dict.items():
                fout.write(str(v) + '\n')
        logging.info("Vocab saved in %s" % output_file)

    @staticmethod
    def _get_default_config():
        c = {
            'unk_token': '<UNK>',
            'sos_token': '<SOS>',
            'eos_token': '<EOS>',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]',
            'add_sos': True,
            'add_eos': True,
            'add_cls': False,  # whether append [CLS] token to the end of the vocab
            'add_sep': False,  # whether append [SEP] token to the end of the vocab
            'add_mask': False,  # whether append [MASK] token to the end of the vocab
        }
        return c
