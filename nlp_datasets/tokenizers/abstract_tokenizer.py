import abc
import logging
import os

import tensorflow as tf


class TokenizerFilter(abc.ABC):

    def drop_token(self, token):
        """Drop this token from vocab file or not.

        Args:
            token: Tokenized word

        Returns:
            A python Boolean. True if drop this word, False else.
        """
        raise NotImplementedError()


class AbstractTokenizer(abc.ABC):
    """Tokenizer for language. The first line of vocab file is always `0   <UNK>`."""

    def __init__(self, config=None):
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config

        self._vocab_size_include_special_tokens = None
        self._init_()

    def _process_line(self, line, token_filters=None):
        """Process line by line in file.

        Args:
            line: A python str, line of file
            token_filters: An iterable of TokenFilter
        """
        raise NotImplementedError()

    def _init_(self):
        self._vocab_size_include_unk = 1  # unk
        self._vocab_size_include_special_tokens = 1  # unk
        self._id2token_dict = {0: self.unk_token}
        self._token2id_dict = {self.unk_token: 0}
        self._id2token_table = None
        self._token2id_table = None

    @property
    def vocab_size(self):
        return self._vocab_size_include_special_tokens

    @property
    def unk_id(self):
        return 0

    @property
    def unk_token(self):
        return self.config.get('unk_token', '<UNK>')

    @property
    def sos_id(self):
        if not self.config['add_sos']:
            return None
        return self._vocab_size_include_unk

    @property
    def sos_token(self):
        if not self.config['add_sos']:
            return None
        return self.config.get('sos_token', '<SOS>')

    @property
    def eos_id(self):
        if not self.config['add_eos']:
            return None
        return self._vocab_size_include_unk + 1

    @property
    def eos_token(self):
        if not self.config['add_eos']:
            return None
        return self.config.get('eos_token', '<EOS>')

    @property
    def cls_id(self):
        if not self.config['add_cls']:
            return None
        return self._vocab_size_include_unk + 2

    @property
    def cls_token(self):
        if not self.config['add_cls']:
            return None
        return self.config.get('cls_token', '[CLS]')

    @property
    def sep_id(self):
        if not self.config['add_sep']:
            return None
        return self._vocab_size_include_unk + 3

    @property
    def sep_token(self):
        if not self.config['add_sep']:
            return None
        return self.config.get('sep_token', '[SEP]')

    @property
    def mask_id(self):
        if not self.config['add_mask']:
            return None
        return self._vocab_size_include_unk + 4

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

    def build_from_corpus(self, corpus_files, token_filters=None):
        """Build lookup table and vocab dict from corpus files.

        Args:
            corpus_files: An iterable of files
            token_filters: An iterable of TokenFilter instances
        """
        self._init_()
        for f in corpus_files:
            if not os.path.exists(f):
                logging.warning('File %s does not exist.' % f)
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip('\n').strip()
                    if not line:
                        continue
                    self._process_line(line, token_filters)

        self._build()

    def build_from_vocab(self, vocab_file):
        """Build lookup table from vocab file. Each line of vocab is `id    token`"""
        self._init_()
        with open(vocab_file, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.strip('\n').strip()
                if not line:
                    continue
                tokens = line.split('\t')
                if len(tokens) != 2:
                    logging.warning('Invalid vocab line: %s' % line)
                    continue
                _id = int(tokens[0])
                token = tokens[1]
                self._id2token_dict[_id] = token
                self._token2id_dict[token] = _id

        assert len(self._token2id_dict.keys()) == len(self._id2token_dict.keys())
        self._vocab_size_include_special_tokens = len(self._token2id_dict.keys())
        # init lookup tables
        self._init_lookup_tables()

    def _build(self):
        assert len(self._token2id_dict.keys()) == len(self._id2token_dict.keys())
        self._vocab_size_include_unk = len(self._token2id_dict.keys())

        # add special tokens
        if self.config['add_sos']:
            self._token2id_dict[self.sos_token] = self.sos_id
            self._id2token_dict[self.sos_id] = self.sos_token
        if self.config['add_eos']:
            self._token2id_dict[self.eos_token] = self.eos_id
            self._id2token_dict[self.eos_id] = self.eos_token
        if self.config['add_cls']:
            self._token2id_dict[self.cls_token] = self.cls_id
            self._id2token_dict[self.cls_id] = self.cls_token
        if self.config['add_sep']:
            self._token2id_dict[self.sep_token] = self.sep_id
            self._id2token_dict[self.sep_id] = self.sep_token
        if self.config['add_mask']:
            self._token2id_dict[self.mask_token] = self.mask_id
            self._id2token_dict[self.mask_id] = self.mask_token

        assert len(self._token2id_dict.keys()) == len(self._id2token_dict.keys())
        self._vocab_size_include_special_tokens = len(self._token2id_dict.keys())
        # init lookup tables
        self._init_lookup_tables()

    def _init_lookup_tables(self):
        token2id_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=list(self._token2id_dict.keys()),
            values=list(self._token2id_dict.values()),
            key_dtype=tf.dtypes.string,
            value_dtype=tf.dtypes.int64)
        self._token2id_table = tf.lookup.StaticHashTable(
            initializer=token2id_initializer,
            default_value=0,  # unk id
            name='token2id_lookup_table')

        id2token_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=list(self._id2token_dict.keys()),
            values=list(self._id2token_dict.values()),
            key_dtype=tf.dtypes.int64,
            value_dtype=tf.dtypes.string)
        self._id2token_table = tf.lookup.StaticHashTable(
            initializer=id2token_initializer,
            default_value=self.config.get('unk_token', '<UNK>'),
            name='id2token_lookup_table')

    def save_to_vocab(self, output_file):
        with open(output_file, mode='wt', encoding='utf8') as fout:
            for k, v in sorted(self._id2token_dict.items(), key=lambda it: it[0]):
                fout.write(str(k) + '\t' + str(v) + '\n')

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
