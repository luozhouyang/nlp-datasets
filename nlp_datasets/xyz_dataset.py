import tensorflow as tf

from nlp_datasets import AbstractXYZDataset


class XYZSameFileDataset(AbstractXYZDataset):

    def _build_dataset(self, files):
        if isinstance(files, list):
            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.flat_map(lambda f: tf.data.TextLineDataset(f).skip(self.config['skip_count']))
        else:
            dataset = tf.data.TextLineDataset(files)
        return dataset

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

    def _build_dataset_from_files(self, files):
        dataset = self._build_dataset(files)
        dataset = dataset.filter(lambda x: tf.size(tf.strings.split([x], sep=self.config['xyz_sep']).values) == 3)
        # split x, y, z in one line
        dataset = dataset.map(
            lambda x: (tf.strings.split([x], sep=self.config['xyz_sep']).values[0],
                       tf.strings.split([x], sep=self.config['xyz_sep']).values[1],
                       tf.strings.split([x], sep=self.config['xyz_sep']).values[2]),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _build_dataset_from_files_for_predict(self, predict_files):
        dataset = self._build_dataset(predict_files)
        dataset = dataset.filter(lambda x: tf.size(tf.strings.split([x], sep=self.config['xyz_sep'])) == 3)
        # split x, y, z in one line, and take x, y only!
        dataset = dataset.map(
            lambda x: (tf.strings.split([x], sep=self.config['xyz_sep']).values[0],
                       tf.strings.split([x], sep=self.config['xyz_sep']).values[1]),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _get_default_config(self):
        parent = super(XYZSameFileDataset, self)._get_default_config()
        parent.update({
            'xyz_sep': '@',
            'skip_count': 0,
        })
        return parent


class XYZSeparateFileDataset(AbstractXYZDataset):

    def _build_dataset(self, files):
        if isinstance(files, list):
            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.flat_map(lambda f: tf.data.TextLineDataset(f).skip(self.config['skip_count']))
        else:
            dataset = tf.data.TextLineDataset(files)
        return dataset

    @tf.function
    def _normalize_z(self, z):
        if tf.equal(z, b'0 1'):
            return tf.constant(1, dtype=tf.dtypes.int64)
        if tf.equal(z, b'1 0'):
            return tf.constant(0, dtype=tf.dtypes.int64)
        if tf.equal(z, b'0'):
            return tf.constant(0, dtype=tf.dtypes.int64)
        if tf.equal(z, b'1'):
            return tf.constant(1, dtype=tf.dtypes.int64)
        return tf.cast(tf.strings.to_number(z), dtype=tf.dtypes.int64)

    def _build_dataset_from_files(self, train_files):
        x_files, y_files, z_files = train_files
        x_dataset = self._build_dataset(x_files)
        y_dataset = self._build_dataset(y_files)
        z_dataset = self._build_dataset(z_files)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset, z_dataset))
        return dataset

    def _build_dataset_from_files_for_predict(self, predict_files):
        x_files, y_files = predict_files
        x_dataset = self._build_dataset(x_files)
        y_dataset = self._build_dataset(y_files)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        return dataset

    def _get_default_config(self):
        parent = super(XYZSeparateFileDataset, self)._get_default_config()
        parent.update({
            'xyz_sep': '@',
            'skip_count': 0,
        })
        return parent
