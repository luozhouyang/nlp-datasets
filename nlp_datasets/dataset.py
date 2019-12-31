import abc

import tensorflow as tf


class Dataset(abc.ABC):

    def build_train_dataset(self, inputs):
        """Build dataset for training.

        Args:
            inputs: file paths

        Returns:
            An instance of ``tf.data.Dataset``
        """
        raise NotImplementedError()

    def build_eval_dataset(self, inputs):
        """Build dataset for evaluation.

        Args:
            inputs: file paths

        Returns:
            An instance of ``tf.data.Dataset``
        """
        raise NotImplementedError()

    def build_predict_dataset(self, inputs):
        """Build dataset for prediction.

        Args:
            inputs: file paths

        Returns:
            An instance of ``tf.data.Dataset``
        """
        raise NotImplementedError()

    def _get_default_config(self):
        """Set default common configurations."""
        config = {
            'buffer_size': 10000000,
            'seed': None,
            'reshuffle_each_iteration': True,
            'prefetch_size': tf.data.experimental.AUTOTUNE,
            'num_parallel_calls': tf.data.experimental.AUTOTUNE,
            'add_sos': True,
            'add_eos': True,
            'skip_count': 0,
            'padding_by_eos': False,
            'drop_remainder': True,
            'bucket_width': 10,
            'train_batch_size': 32,
            'eval_batch_size': 32,
            'predict_batch_size': 32,
            'repeat': 1,
        }
        return config
