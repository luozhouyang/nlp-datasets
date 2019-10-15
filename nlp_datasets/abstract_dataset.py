import abc

import tensorflow as tf


class AbstractDataset(abc.ABC):
    """Dataset interface."""

    def __init__(self, x_tokenizer, y_tokenizer, config=None):
        """Constructor.

        Args:
            x_tokenizer: An instance of AbstractTokenizer for x language.
            y_tokenizer: An instance of AbstractTokenizer for y language.
            config: A python dict.
        """
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config

        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer

    def build_train_dataset(self, train_files):
        """Build dataset for training."""
        raise NotImplementedError()

    def build_eval_dataset(self, eval_files):
        """Build dataset for evaluation."""
        raise NotImplementedError()

    def build_predict_dataset(self, predict_files):
        """Build dataset for prediction."""
        raise NotImplementedError()

    @staticmethod
    def _get_default_config():
        c = {
            'add_sos': True,
            'add_eos': True,
            'padding_by_eos': False,  # padding sequence by eos_token if True, unk_token else.
            'drop_remainder': False,
        }
        return c


class AbstractXYDataset(AbstractDataset):
    """Dataset interface for x-y sequence pair models."""

    def _shuffle_dataset(self, dataset):
        """Shuffle dataset."""
        dataset = dataset.shuffle(
            buffer_size=self.config['buffer_size'],  # buffer_size must be greater than total num of examples
            seed=self.config['seed'],
            reshuffle_each_iteration=self.config['reshuffle_each_iteration'])
        return dataset

    def _split_line(self, dataset):
        """Split sequence to list."""
        dataset = dataset.map(
            lambda x, y: (
                tf.strings.split([x], sep=self.config['sequence_sep']).values,
                tf.strings.split([y], sep=self.config['sequence_sep']).values),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _split_line_for_predict(self, dataset):
        dataset = dataset.map(
            lambda x: tf.strings.split([x], sep=self.config['sequence_sep']).values,
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _filter_dataset(self, dataset):
        """Filter examples that are empty or too long."""
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        x_max_len = self.config['x_max_len']
        if x_max_len > 0:
            dataset = dataset.filter(lambda x, y: tf.size(x) <= x_max_len)
        y_max_len = self.config['y_max_len']
        if y_max_len > 0:
            dataset = dataset.filter(lambda x, y: tf.size(y) <= y_max_len)
        return dataset

    def _filter_dataset_for_predict(self, dataset):
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        x_max_len = self.config['x_max_len']
        if x_max_len > 0:
            dataset = dataset.filter(lambda x, y: tf.size(x) <= x_max_len)
        return dataset

    def _convert_tokens_to_ids(self, dataset):
        """Convert string tokens to ids."""
        dataset = dataset.map(
            lambda x, y: (self.x_tokenizer.encode(x), self.y_tokenizer.encode(y)),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _convert_tokens_to_ids_for_predict(self, dataset):
        dataset = dataset.map(
            lambda x: self.x_tokenizer.encode(x),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _add_special_tokens(self, dataset):
        """Attempt to add sos and eos tokens."""
        if self.config['add_sos']:
            x_sos_id = tf.constant(self.x_tokenizer.sos_id, dtype=tf.dtypes.int64)
            y_sos_id = tf.constant(self.y_tokenizer.sos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x, y: (tf.concat(([x_sos_id], x), axis=0), tf.concat(([y_sos_id], y), axis=0)),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        if self.config['add_eos']:
            x_eos_id = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
            y_eos_id = tf.constant(self.y_tokenizer.eos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x, y: (tf.concat((x, [x_eos_id]), axis=0), tf.concat((y, [y_eos_id]), axis=0)),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        return dataset

    def _add_special_tokens_for_predict(self, dataset):
        if self.config['add_sos']:
            x_sos_id = tf.constant(self.x_tokenizer.sos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x,: tf.concat(([x_sos_id], x), axis=0),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        if self.config['add_eos']:
            x_eos_id = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x: tf.concat((x, [x_eos_id]), axis=0),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        return dataset

    def _padding_and_batching(self, dataset, batch_size):
        x_max_len = self.config['x_max_len']
        x_padded_shape = x_max_len if x_max_len > 0 else None
        y_max_len = self.config['y_max_len']
        y_padded_shape = y_max_len if y_max_len > 0 else None
        if self.config['padding_by_eos']:
            x_padding_value = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
            y_padding_value = tf.constant(self.y_tokenizer.eos_id, dtype=tf.dtypes.int64)
        else:
            x_padding_value = tf.constant(self.x_tokenizer.unk_id, dtype=tf.dtypes.int64)
            y_padding_value = tf.constant(self.y_tokenizer.unk_id, dtype=tf.dtypes.int64)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([x_padded_shape], [y_padded_shape]),
            padding_values=(x_padding_value, y_padding_value),
            drop_remainder=self.config['drop_remainder'])
        return dataset

    def _padding_and_batching_for_predict(self, dataset, batch_size):
        x_max_len = self.config['x_max_len']
        x_padded_shape = x_max_len if x_max_len > 0 else None
        if self.config['padding_by_eos']:
            x_padding_value = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
        else:
            x_padding_value = tf.constant(self.x_tokenizer.unk_id, dtype=tf.dtypes.int64)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=[x_padded_shape],
            padding_values=x_padding_value)
        return dataset

    def build_train_dataset(self, train_files):
        """Build dataset for training.

        Args:
            train_files: An iterable of tuple (x_file, y_file)

        Returns:
            A tf.data.Dataset object
        """
        dataset = self._build_dataset_from_files(train_files)
        dataset = self._shuffle_dataset(dataset)
        dataset = self._split_line(dataset)
        dataset = self._filter_dataset(dataset)
        dataset = self._convert_tokens_to_ids(dataset)
        dataset = self._add_special_tokens(dataset)
        batch_size = self.config['train_batch_size']
        dataset = self._padding_and_batching(dataset, batch_size)
        return dataset

    def build_eval_dataset(self, eval_files):
        """Build dataset for evaluation.

        Args:
            eval_files: An iterable of tuple (x_file, y_file)

        Returns:
            A tf.data.Dataset object
        """
        dataset = self._build_dataset_from_files(eval_files)
        dataset = self._shuffle_dataset(dataset)
        dataset = self._split_line(dataset)
        dataset = self._filter_dataset(dataset)
        dataset = self._convert_tokens_to_ids(dataset)
        dataset = self._add_special_tokens(dataset)
        batch_size = self.config['eval_batch_size']
        dataset = self._padding_and_batching(dataset, batch_size)
        return dataset

    def build_predict_dataset(self, predict_files):
        """Build dataset for prediction.

        Args:
            predict_files: An iterable of x_file

        Returns:
            An tf.data.Dataset object
        """
        dataset = self._build_dataset_from_files_for_predict(predict_files)
        dataset = self._split_line_for_predict(dataset)
        dataset = self._convert_tokens_to_ids_for_predict(dataset)
        dataset = self._add_special_tokens_for_predict(dataset)
        dataset = self._padding_and_batching_for_predict(dataset, self.config['predict_batch_size'])
        return dataset

    def _build_dataset_from_files(self, files):
        """Build dataset from file(s) for training or evaluation mode.

        Args:
            files: An iterable of tuple (x_file, y_file)

        Returns:
            A tf.data.Dataset object
        """
        raise NotImplementedError()

    def _build_dataset_from_files_for_predict(self, files):
        """Build dataset from file(s) for prediction mode.

        Args:
            files: An iterable of x_file

        Returns:
            A tf.data.Dataset object
        """
        raise NotImplementedError()

    def _get_default_config(self):
        parent = super(AbstractXYDataset, self)._get_default_config()
        parent.update({
            'x_max_len': -1,
            'y_max_len': -1,
            'buffer_size': 10000000,
            'seed': None,
            'reshuffle_each_iteration': True,
            'sequence_sep': ' ',  # split x and y to list
            'prefetch_size': tf.data.experimental.AUTOTUNE,
            'num_parallel_calls': tf.data.experimental.AUTOTUNE,
            'train_batch_size': 32,
            'eval_batch_size': 32,
            'predict_batch_size': 32,
        })
        return parent


class AbstractXYZDataset(AbstractDataset):

    def _shuffle_dataset(self, dataset):
        """Shuffle dataset."""
        dataset = dataset.shuffle(
            buffer_size=self.config['buffer_size'],  # buffer_size must be greater than total num of examples
            seed=self.config['seed'],
            reshuffle_each_iteration=self.config['reshuffle_each_iteration'])
        return dataset

    def _split_line(self, dataset):
        dataset = dataset.map(
            lambda x, y, z: (tf.strings.split([x], sep=self.config['sequence_sep']).values,
                             tf.strings.split([y], sep=self.config['sequence_sep']).values,
                             self._normalize_z(z)),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    @tf.function
    def _normalize_z(self, z):
        raise NotImplementedError()

    def _split_line_for_predict(self, dataset):
        dataset = dataset.map(
            lambda x, y: (tf.strings.split([x], sep=self.config['sequence_sep']).values,
                          tf.strings.split([y], sep=self.config['sequence_sep']).values,),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _filter_dataset(self, dataset):
        dataset = dataset.filter(lambda x, y, z: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        x_max_len = self.config['x_max_len']
        if x_max_len > 0:
            dataset = dataset.filter(lambda x, y, z: (tf.size(x) <= x_max_len))
        y_max_len = self.config['y_max_len']
        if y_max_len > 0:
            dataset = dataset.filter(lambda x, y, z: (tf.size(y) <= y_max_len))
        return dataset

    def _convert_tokens_to_ids(self, dataset):
        dataset = dataset.map(
            lambda x, y, z: (self.x_tokenizer.encode(x), self.y_tokenizer.encode(y), z),
            num_parallel_calls=self.config['num_parallel_calls'],
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _convert_tokens_to_ids_for_predict(self, dataset):
        dataset = dataset.map(
            lambda x, y: (self.x_tokenizer.encode(x), self.y_tokenizer.encode(y)),
            num_parallel_calls=self.config['num_parallel_calls'],
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _add_special_tokens(self, dataset):
        if self.config['add_sos']:
            x_sos = tf.constant(self.x_tokenizer.sos_id, dtype=tf.dtypes.int64)
            y_sos = tf.constant(self.y_tokenizer.sos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x, y, z: (tf.concat(([x_sos], x), axis=0),
                                 tf.concat(([y_sos], y), axis=0),
                                 z),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        if self.config['add_eos']:
            x_eos = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
            y_eos = tf.constant(self.y_tokenizer.eos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x, y, z: (tf.concat((x, [x_eos]), axis=0),
                                 tf.concat((y, [y_eos]), axis=0),
                                 z),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        return dataset

    def _add_special_tokens_for_predict(self, dataset):
        if self.config['add_sos']:
            x_sos = tf.constant(self.x_tokenizer.sos_id, dtype=tf.dtypes.int64)
            y_sos = tf.constant(self.y_tokenizer.sos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x, y: (tf.concat(([x_sos], x), axis=0),
                              tf.concat(([y_sos], y), axis=0),),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        if self.config['add_eos']:
            x_eos = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
            y_eos = tf.constant(self.y_tokenizer.eos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x, y: (tf.concat((x, [x_eos]), axis=0),
                              tf.concat((y, [y_eos]), axis=0),),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        return dataset

    def _padding_and_batching(self, dataset, batch_size):
        x_max_len = self.config['x_max_len']
        x_padded_shape = x_max_len if x_max_len > 0 else None
        y_max_len = self.config['y_max_len']
        y_padded_shape = y_max_len if y_max_len > 0 else None
        # padding sequence by sos or unk
        if self.config['padding_by_eos']:
            x_padding_value = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
            y_padding_value = tf.constant(self.y_tokenizer.eos_id, dtype=tf.dtypes.int64)
        else:
            x_padding_value = tf.constant(self.x_tokenizer.unk_id, dtype=tf.dtypes.int64)
            y_padding_value = tf.constant(self.y_tokenizer.unk_id, dtype=tf.dtypes.int64)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padding_values=(x_padding_value, y_padding_value, tf.constant(0, dtype=tf.dtypes.int64)),
            padded_shapes=([x_padded_shape], [y_padded_shape], []),
            drop_remainder=self.config['drop_remainder']
        ).prefetch(self.config['prefetch_size'])

        # map (x,y,z) to ((x,y),z) cause (x,y) are two inputs of model.
        dataset = dataset.map(
            lambda x, y, z: ((x, y), z),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _padding_and_batching_for_predict(self, dataset, batch_size):
        x_max_len = self.config['x_max_len']
        x_padded_shape = x_max_len if x_max_len > 0 else None
        y_max_len = self.config['y_max_len']
        y_padded_shape = y_max_len if y_max_len > 0 else None
        if self.config['padding_by_eos']:
            x_padding_value = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
            y_padding_value = tf.constant(self.y_tokenizer.eos_id, dtype=tf.dtypes.int64)
        else:
            x_padding_value = tf.constant(self.x_tokenizer.unk_id, dtype=tf.dtypes.int64)
            y_padding_value = tf.constant(self.y_tokenizer.unk_id, dtype=tf.dtypes.int64)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padding_values=(x_padding_value, y_padding_value),
            padded_shapes=([x_padded_shape], [y_padded_shape])
        ).prefetch(self.config['prefetch_size'])

        # map (x,y) to ((x,y)) cause (x,y) are two inputs of model.
        dataset = dataset.map(
            lambda x, y: ((x, y),),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def build_train_dataset(self, train_files):
        dataset = self._build_dataset_from_files(train_files)
        dataset = self._shuffle_dataset(dataset)
        dataset = self._split_line(dataset)
        dataset = self._filter_dataset(dataset)
        dataset = self._convert_tokens_to_ids(dataset)
        dataset = self._add_special_tokens(dataset)
        dataset = self._padding_and_batching(dataset, self.config['train_batch_size'])
        return dataset

    def build_eval_dataset(self, eval_files):
        dataset = self._build_dataset_from_files(eval_files)
        dataset = self._shuffle_dataset(dataset)
        dataset = self._split_line(dataset)
        dataset = self._filter_dataset(dataset)
        dataset = self._convert_tokens_to_ids(dataset)
        dataset = self._add_special_tokens(dataset)
        dataset = self._padding_and_batching(dataset, self.config['eval_batch_size'])
        return dataset

    def build_predict_dataset(self, predict_files):
        dataset = self._build_dataset_from_files_for_predict(predict_files)
        dataset = self._split_line_for_predict(dataset)
        dataset = self._convert_tokens_to_ids_for_predict(dataset)
        dataset = self._add_special_tokens_for_predict(dataset)
        dataset = self._padding_and_batching_for_predict(dataset, self.config['predict_batch_size'])
        return dataset

    def _build_dataset_from_files(self, files):
        raise NotImplementedError()

    def _build_dataset_from_files_for_predict(self, files):
        raise NotImplementedError()

    def _get_default_config(self):
        parent = super(AbstractXYZDataset, self)._get_default_config()
        parent.update({
            'x_max_len': -1,
            'y_max_len': -1,
            'buffer_size': 10000000,
            'seed': None,
            'reshuffle_each_iteration': True,
            'sequence_sep': ' ',
            'prefetch_size': tf.data.experimental.AUTOTUNE,
            'num_parallel_calls': tf.data.experimental.AUTOTUNE,
            'train_batch_size': 32,
            'eval_batch_size': 32,
            'predict_batch_size': 32,
        })
        return parent
