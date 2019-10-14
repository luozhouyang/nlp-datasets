import unittest

import tensorflow as tf

from nlp_datasets.tokenizers import EmptyTokenFilter
from nlp_datasets.tokenizers import SpaceTokenizer, XYSpaceTokenizer, XYZSpaceTokenizer
from nlp_datasets.utils import data_dir_utils


class SpaceTokenizerTest(unittest.TestCase):

    def buildTokenizer(self):
        tokenizer = SpaceTokenizer()
        corpus = ['iwslt15.tst2013.100.en']
        corpus = [data_dir_utils.get_data_file(f) for f in corpus]
        tokenizer.build_from_corpus(corpus, token_filters=[EmptyTokenFilter()])
        return tokenizer

    def testTokenize(self):
        tokenizer = self.buildTokenizer()
        print(tokenizer.token2id_dict)
        print(tokenizer.id2token_dict)
        print(tokenizer.vocab_size)

    def testConvertTokens2Ids(self):
        tokenizer = self.buildTokenizer()
        print('token2 id dict: ', tokenizer.token2id_dict)
        words = tf.constant(['I', 'am', 'a', 'developer'])
        v = tokenizer.encode(words)
        print(v)

    def testConvertIds2Tokens(self):
        tokenizer = self.buildTokenizer()
        print('id2token dict: ', tokenizer.id2token_dict)
        ids = tf.constant([1, 0, 2, 3, 4], dtype=tf.dtypes.int64)
        v = tokenizer.decode(ids)
        print(v)

    def testSaveVocabFile(self):
        tokenizer = self.buildTokenizer()
        tokenizer.save_to_vocab(data_dir_utils.get_data_file('vocab.test.txt'))
        print(tokenizer.token2id_dict)
        print(tokenizer.id2token_dict)

    def testBuildFromVocab(self):
        print('============start build from vocab=============')
        tokenizer = SpaceTokenizer()
        tokenizer.build_from_vocab(data_dir_utils.get_data_file('vocab.test.txt'))
        print('token2id dict: ', tokenizer.token2id_dict)
        print('id2token dict: ', tokenizer.id2token_dict)
        words = tf.constant(['I', 'am', 'a', 'developer'])
        v0 = tokenizer.encode(words)
        print(v0)
        ids = tf.constant([1, 0, 2, 3, 4], dtype=tf.dtypes.int64)
        v1 = tokenizer.decode(ids)
        print(v1)
        print('============end build from vocab=============')


class XYSpaceTokenizerTest(SpaceTokenizerTest):

    def buildTokenizer(self):
        tokenizer = XYSpaceTokenizer()
        corpus = ['iwslt15.tst2013.100.en']
        corpus = [data_dir_utils.get_data_file(f) for f in corpus]
        tokenizer.build_from_corpus(corpus, max_vocab_size=200, token_filters=[EmptyTokenFilter()])
        print('sos id: ', tokenizer.sos_id)
        print('eos id: ', tokenizer.eos_id)
        print('vocab size: ', tokenizer.vocab_size)
        return tokenizer


class XYZSpaceTokenizerTest(SpaceTokenizerTest):

    def buildTokenizer(self):
        tokenizer = XYZSpaceTokenizer()
        corpus = ['iwslt15.tst2013.100.en']
        corpus = [data_dir_utils.get_data_file(f) for f in corpus]
        tokenizer.build_from_corpus(corpus, token_filters=[EmptyTokenFilter()])
        print('sos id: ', tokenizer.sos_id)
        print('eos id: ', tokenizer.eos_id)
        print('vocab size: ', tokenizer.vocab_size)
        return tokenizer


if __name__ == '__main__':
    unittest.main()
