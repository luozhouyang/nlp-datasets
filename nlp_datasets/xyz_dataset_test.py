from nlp_datasets import XYZSameFileDataset, XYZSeparateFileDataset

from nlp_datasets.utils import data_dir_utils
import unittest
from nlp_datasets.tokenizers import SpaceTokenizer


class XYZDatasetTest(unittest.TestCase):

    @staticmethod
    def buildXTokenizer():
        t = SpaceTokenizer()
        t.build_from_corpus([data_dir_utils.get_data_file('dssm.query.txt')])
        return t

    @staticmethod
    def buildYTokenizer():
        t = SpaceTokenizer()
        t.build_from_corpus([data_dir_utils.get_data_file('dssm.doc.txt')])
        return t

    def testXYZSameFileDataset(self):
        x_tokenizer = self.buildXTokenizer()
        y_tokenizer = self.buildYTokenizer()
        config = {
            'xyz_sep': '@',
            'sequence_sep': ' ',
            'x_max_len': -1,
            'y_max_len': -1,
            'train_batch_size': 2,
            'eval_batch_size': 2,
            'predict_batch_size': 1
        }
        ds = XYZSameFileDataset(x_tokenizer, y_tokenizer, config=config)
        files = [data_dir_utils.get_data_file('dssm.query.doc.label.txt')]
        train_dataset = ds.build_train_dataset(files)
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

    def testXYZSeparateFileDataset(self):
        x_tokenizer = self.buildXTokenizer()
        y_tokenizer = self.buildYTokenizer()
        config = {
            'xyz_sep': '@',
            'sequence_sep': ' ',
            'x_max_len': -1,
            'y_max_len': -1,
            'train_batch_size': 2,
            'eval_batch_size': 2,
            'predict_batch_size': 1
        }
        ds = XYZSeparateFileDataset(x_tokenizer, y_tokenizer, config=config)
        files = ([data_dir_utils.get_data_file('dssm.query.txt')],
                 [data_dir_utils.get_data_file('dssm.doc.txt')],
                 [data_dir_utils.get_data_file('dssm.label.txt')])
        train_dataset = ds.build_train_dataset(files)
        v = next(iter(train_dataset))
        print(v)
        print('=' * 50)
        eval_dataset = ds.build_eval_dataset(files)
        v = next(iter(eval_dataset))
        print(v)
        print('=' * 50)
        predict_dataset = ds.build_predict_dataset(
            ([data_dir_utils.get_data_file('dssm.query.txt')], [data_dir_utils.get_data_file('dssm.doc.txt')]))
        v = next(iter(predict_dataset))
        print(v)


if __name__ == '__main__':
    unittest.main()
