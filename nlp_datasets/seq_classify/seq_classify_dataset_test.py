import tensorflow as tf
from nlp_datasets.utils import data_dir_utils as utils
from nlp_datasets.seq_classify import SeqClassifyDataset
from nlp_datasets.tokenizers import SpaceTokenizer


class SeqClassifyDatasetTest(tf.test.TestCase):

    def testBuildDatasetFromSameFile(self):
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

    def testBuildDatasetFromSeparateFile(self):
        files = [
            (utils.get_data_file('classify.seq.txt'), utils.get_data_file('classify.label.txt'))
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


if __name__ == '__main__':
    tf.test.main()
