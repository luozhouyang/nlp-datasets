import tensorflow as tf

from nlp_datasets.utils import data_dir_utils as utils
from nlp_datasets.seq2seq.seq2seq_dataset import Seq2SeqDataset
from nlp_datasets.tokenizers import SpaceTokenizer


class Seq2SeqDatasetTest(tf.test.TestCase):

    def testBuildDatasetFromSameFile(self):
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

    def testBuildDatasetFromSeparateFile(self):
        files = [
            (utils.get_data_file('iwslt15.tst2013.100.en'), utils.get_data_file('iwslt15.tst2013.100.vi')),
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

        predict_files = [utils.get_data_file('iwslt15.tst2013.100.en')]
        predict_dataset = dataset.build_predict_dataset(predict_files)
        print(next(iter(predict_dataset)))
        print('=' * 120)


if __name__ == '__main__':
    tf.test.main()
