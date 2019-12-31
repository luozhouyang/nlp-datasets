import tensorflow as tf

from nlp_datasets.dataset import Dataset


class SeqMatchDataset(Dataset):

    def __init__(self, x_tokenizer, y_tokenizer, config=None):
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config

    def _build_dataset_from_files(self, files):
        if not files:
            raise ValueError('Argument `files` must not be empty or None.')
        if all(isinstance(f, str) for f in files):  # files: [file_0, file_1, ...]
            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(self.config['skip_count']))
            # filter invalid data
            dataset = dataset.filter(
                lambda x: tf.equal(3, tf.size(tf.strings.split([x], sep=self.config['xyz_sep']).values)))
            dataset = dataset.map(
                lambda x: (tf.strings.split([x], sep=self.config['xyz_sep']).values[0],
                           tf.strings.split([x], sep=self.config['xyz_sep']).values[1],
                           tf.strings.split([x], sep=self.config['xyz_sep']).values[2]),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])

            if self.config['repeat']:
                dataset = dataset.repeat(self.config['repeat'])
            return dataset
        # files: [(x_file_0, y_file_0, z_file_0), (x_file_1, y_file_1, z_file_1), ...]
        if all(isinstance(f, tuple) for f in files):
            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.filter(lambda x: tf.equal(3, tf.size(x)))
            dataset = dataset.map(lambda x: (x[0], x[1], x[2]))
            dataset = dataset.flat_map(
                lambda x, y, z: tf.data.Dataset.zip(
                    (tf.data.TextLineDataset(x).skip(self.config['skip_count']),
                     tf.data.TextLineDataset(y).skip(self.config['skip_count']),
                     tf.data.TextLineDataset(z).skip(self.config['skip_count'])))
            ).prefetch(self.config['prefetch_size'])
            if self.config['repeat']:
                dataset = dataset.repeat(self.config['repeat'])
            return dataset
        raise ValueError(
            r"Argument `files` must be one of the following formats:"
            " 1. [file_0, file_1, ...] if seq_x, seq_y and label_z are in the same file."
            " 2. [(x_file_0, y_file_0, z_file_0), (x_file_1, y_file_1, z_file_1), ...] if seq_x, seq_y and label_z are in separate files.")

    def _shuffle_dataset(self, dataset):
        return dataset.shuffle(
            buffer_size=self.config['buffer_size'],
            seed=self.config['seed'],
            reshuffle_each_iteration=self.config['reshuffle_each_iteration'])

    @tf.function
    def _normalize_z(self, z):
        if tf.equal(z, '0 1'):
            return tf.constant(1, dtype=tf.dtypes.int64)
        if tf.equal(z, '1 0'):
            return tf.constant(0, dtype=tf.dtypes.int64)
        if tf.equal(z, '0'):
            return tf.constant(0, dtype=tf.dtypes.int64)
        if tf.equal(z, '1'):
            return tf.constant(1, dtype=tf.dtypes.int64)
        return tf.cast(tf.strings.to_number(z), dtype=tf.dtypes.int64)

    def _split_sequence(self, dataset):
        dataset = dataset.map(
            lambda x, y, z: (tf.strings.split([x], sep=self.config['sep']).values,
                             tf.strings.split([y], sep=self.config['sep']).values,
                             self._normalize_z(z)),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _filter_sequence(self, dataset):
        dataset = dataset.filter(lambda x, y, z: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        x_max_len = self.config['x_max_len']
        if x_max_len > 0:
            dataset = dataset.filter(lambda x, y, z: tf.size(x) <= x_max_len)
        y_max_len = self.config['y_max_len']
        if y_max_len > 0:
            dataset = dataset.filter(lambda x, y, z: tf.size(y) <= y_max_len)
        return dataset

    def _convert_tokens_to_ids(self, dataset):
        dataset = dataset.map(
            lambda x, y, z: (self.x_tokenizer.encode(x), self.y_tokenizer.encode(y), z),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _add_special_tokens(self, dataset):
        if self.config['add_sos']:
            x_sos_id = tf.constant(self.x_tokenizer.sos_id, dtype=tf.dtypes.int64)
            y_sos_id = tf.constant(self.y_tokenizer.sos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x, y, z: (tf.concat(([x_sos_id], x), axis=0), tf.concat(([y_sos_id], y), axis=0), z),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        if self.config['add_eos']:
            x_eos_id = tf.constant(self.x_tokenizer.eos_id, dtype=tf.dtypes.int64)
            y_eos_id = tf.constant(self.y_tokenizer.eos_id, dtype=tf.dtypes.int64)
            dataset = dataset.map(
                lambda x, y, z: (tf.concat((x, [x_eos_id]), axis=0), tf.concat((y, [y_eos_id]), axis=0), z),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        return dataset

    def _padding_and_batching(self, dataset, batch_size, bucket_width):
        # batching
        def _batch_fn(x):
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

            x = x.padded_batch(
                batch_size=batch_size,
                padded_shapes=([x_padded_shape], [y_padded_shape], []),
                padding_values=(x_padding_value, y_padding_value, tf.constant(0, dtype=tf.int64)),
                drop_remainder=self.config['drop_remainder'])
            return x

        def _key_fn(x, y, z):
            # x -> src sequence, y -> target sequence, corresponding to dataset's two element
            bucket_id = tf.maximum(tf.size(x) // bucket_width, tf.size(y) // bucket_width)
            return tf.cast(bucket_id, tf.dtypes.int64)

        def _reduce_fn(k, x):
            # k -> unused key, x -> dataset
            return _batch_fn(x)

        bucket_fn = tf.data.experimental.group_by_window(
            key_func=_key_fn, reduce_func=_reduce_fn, window_size=batch_size)
        return dataset.apply(bucket_fn)

    def build_train_dataset(self, inputs):
        dataset = self._build_dataset_from_files(inputs)
        dataset = self._shuffle_dataset(dataset)
        dataset = self._split_sequence(dataset)
        dataset = self._filter_sequence(dataset)
        dataset = self._convert_tokens_to_ids(dataset)
        dataset = self._add_special_tokens(dataset)
        batch_size = self.config['train_batch_size']
        bucket_width = self.config['bucket_width']
        dataset = self._padding_and_batching(dataset, batch_size, bucket_width)
        return dataset

    def build_eval_dataset(self, inputs):
        dataset = self._build_dataset_from_files(inputs)
        dataset = self._shuffle_dataset(dataset)
        dataset = self._split_sequence(dataset)
        dataset = self._filter_sequence(dataset)
        dataset = self._convert_tokens_to_ids(dataset)
        dataset = self._add_special_tokens(dataset)
        batch_size = self.config['eval_batch_size']
        bucket_width = self.config['bucket_width']
        dataset = self._padding_and_batching(dataset, batch_size, bucket_width)
        return dataset

    def build_predict_dataset(self, inputs):
        def _build_dataset_from_files(files):
            if not files:
                raise ValueError("Argument `files` must not be empty or None.")
            if all(isinstance(f, str) for f in files):  # files: [file_0, file_1, ...]
                d = tf.data.Dataset.from_tensor_slices(files)
                d = d.flat_map(lambda x: tf.data.TextLineDataset(x).skip(self.config['skip_count']))
                d = d.filter(
                    lambda x: tf.greater(tf.size(tf.strings.split([x], sep=self.config['xyz_sep']).values), 2))
                d = d.map(
                    # only take features, not labels
                    lambda x: (tf.strings.split([x], sep=self.config['xyz_sep']).values[0],
                               tf.strings.split([x], sep=self.config['xyz_sep']).values[1]),
                    num_parallel_calls=self.config['num_parallel_calls'])
                return d
            if all(isinstance(f, tuple) for f in files):  # files: [(x_file_0, y_file_0), (x_file_1, y_file_1,), ...]
                d = tf.data.Dataset.from_tensor_slices(files)
                d = d.filter(lambda x: tf.equal(2, tf.size(x)))
                d = d.map(lambda x: (x[0], x[1]))
                d = d.flat_map(
                    lambda x, y: tf.data.Dataset.zip(
                        (tf.data.TextLineDataset(x).skip(self.config['skip_count']),
                         tf.data.TextLineDataset(y).skip(self.config['skip_count']))
                    ))
                return d
            raise ValueError(
                r"Argument `files` must be one of the following formats:"
                " 1. [file_0, file_1, ...] if seq_x and seq_y are in the same file."
                " 2. [(x_file_0, y_file_0), (x_file_1, y_file_1), ...] if seq_x and seq_y are in separate files.")

        def _split_sequence(d):
            return d.map(
                lambda x, y: (tf.strings.split([x], sep=self.config['sep']).values,
                              tf.strings.split([y], sep=self.config['sep']).values),
                num_parallel_calls=self.config['num_parallel_calls'])

        def _convert_tokens_to_ids(d):
            return d.map(lambda x, y: (self.x_tokenizer.encode(x), self.y_tokenizer.encode(y)))

        def _add_special_tokens(d):
            if self.config['add_sos']:
                sos_id = tf.constant(self.x_tokenizer.sos_id, dtype=tf.int64)
                d = d.map(lambda x, y: (tf.concat(([sos_id], x), axis=0), tf.concat(([sos_id], y), axis=0)),
                          num_parallel_calls=self.config['num_parallel_calls'])
            if self.config['add_eos']:
                eos_id = tf.constant(self.x_tokenizer.eos_id, dtype=tf.int64)
                d = d.map(lambda x, y: (tf.concat((x, [eos_id]), axis=0), tf.concat((y, [eos_id]), axis=0)),
                          num_parallel_calls=self.config['num_parallel_calls'])
            return d

        def _padding_and_batching(x):
            x_padded_shape = self.config['x_max_len'] if self.config['x_max_len'] > 0 else None
            y_padded_shape = self.config['y_max_len'] if self.config['y_max_len'] > 0 else None
            x_padding_value = self.x_tokenizer.eos_id if self.config['padding_by_eos'] else self.x_tokenizer.unk_id
            x_padding_value = tf.constant(x_padding_value, dtype=tf.int64)
            y_padding_value = self.y_tokenizer.eos_id if self.config['padding_by_eos'] else self.y_tokenizer.unk_id
            y_padding_value = tf.constant(y_padding_value, dtype=tf.int64)
            x = x.padded_batch(
                batch_size=self.config['predict_batch_size'],
                padded_shapes=([x_padded_shape], [y_padded_shape]),
                padding_values=(x_padding_value, y_padding_value),
                drop_remainder=self.config['drop_remainder'])
            return x

        dataset = _build_dataset_from_files(inputs)
        dataset = _split_sequence(dataset)
        dataset = _convert_tokens_to_ids(dataset)
        dataset = _add_special_tokens(dataset)
        dataset = _padding_and_batching(dataset)
        return dataset

    def _get_default_config(self):
        base = super(SeqMatchDataset, self)._get_default_config()
        config = {
            'xyz_sep': '@',
            'sep': ' ',
            'x_max_len': -1,
            'y_max_len': -1,
        }
        return dict(list(base.items()) + list(config.items()))
