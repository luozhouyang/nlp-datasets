from nlp_datasets import XYSameFileDataset, XYSeparateFileDataset
import unittest
from nlp_datasets.utils import data_dir_utils
from nlp_datasets.tokenizers import SpaceTokenizer


class XYDatasetTest(unittest.TestCase):

    @staticmethod
    def buildXTokenizer():
        t = SpaceTokenizer()
        t.build_from_corpus([data_dir_utils.get_data_file('iwslt15.tst2013.100.en')])
        return t

    @staticmethod
    def buildYTokenizer():
        t = SpaceTokenizer()
        t.build_from_corpus([data_dir_utils.get_data_file('iwslt15.tst2013.100.vi')])
        return t

    def testXYSameFileDataset(self):
        x_tokenizer = self.buildXTokenizer()
        y_tokenizer = self.buildYTokenizer()
        config = {
            'xy_sep': '@',
            'sequence_sep': ' ',
            'x_max_len': -1,
            'y_max_len': -1,
            'repeat': 10,
        }
        ds = XYSameFileDataset(x_tokenizer, y_tokenizer, config=config)
        files = [data_dir_utils.get_data_file('iwslt15.tst2013.100.envi')]
        train_dataset = ds.build_train_dataset(train_files=files)
        v = next(iter(train_dataset))
        print(v)
        print('=' * 50)
        eval_dataset = ds.build_eval_dataset(files)
        v = next(iter(eval_dataset))
        print(v)
        print('=' * 50)
        predict_dataset = ds.build_predict_dataset(files)
        v = next(iter(predict_dataset))
        print(v)

    def testXYSeparateFileDataset(self):
        x_tokenizer = self.buildXTokenizer()
        y_tokenizer = self.buildYTokenizer()
        config = {
            'sequence_sep': ' ',
            'x_max_len': -1,
            'y_max_len': -1,
            'repeat': 10,
        }
        ds = XYSeparateFileDataset(x_tokenizer, y_tokenizer, config)
        files = ([data_dir_utils.get_data_file('iwslt15.tst2013.100.en')],
                 [data_dir_utils.get_data_file('iwslt15.tst2013.100.vi')])
        train_dataset = ds.build_train_dataset(train_files=files)
        v = next(iter(train_dataset))
        print(v)
        print('=' * 50)
        eval_dataset = ds.build_eval_dataset(files)
        v = next(iter(eval_dataset))
        print(v)
        print('=' * 50)
        predict_dataset = ds.build_predict_dataset([data_dir_utils.get_data_file('iwslt15.tst2013.100.en')])
        v = next(iter(predict_dataset))
        print(v)


if __name__ == '__main__':
    unittest.main()
