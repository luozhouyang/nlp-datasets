import tensorflow as tf

from nlp_datasets import AbstractXYDataset


class XYSameFileDataset(AbstractXYDataset):
    """Build dataset for examples that x and y are in the same file."""

    def _build_dataset(self, files):
        if isinstance(files, list):
            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.flat_map(lambda f: tf.data.TextLineDataset(f).skip(self.config['skip_count']))
        else:
            dataset = tf.data.TextLineDataset(files)
        # repeat the dataset
        if self.config['repeat']:
            dataset = dataset.repeat(self.config['repeat'])
        return dataset

    def _build_dataset_from_files(self, files):
        """Build dataset from file(s).

        Args:
            files: A file or a list of file.

        Returns:
            A tf.data.Dataset object
        """
        dataset = self._build_dataset(files)
        # split x, y in one line
        dataset = dataset.map(
            lambda x: (tf.strings.split([x], sep=self.config['xy_sep']).values[0],
                       tf.strings.split([x], sep=self.config['xy_sep']).values[1]),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _build_dataset_from_files_for_predict(self, files):
        """Build dataset from file(s) for prediction.

        Args:
            files: A file or a list of file.

        Returns:
            A tf.data.Dataset object
        """
        dataset = self._build_dataset(files)
        dataset = dataset.map(
            lambda x: (tf.strings.split([x], sep=self.config['xy_sep']).values[0]),  # only takes x
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _get_default_config(self):
        parent = super(XYSameFileDataset, self)._get_default_config()
        parent.update({
            'xy_sep': '@',  # split x and y
            'skip_count': 0,  # may useful when need to skip head line in files,
            'repeat': 1,  # repeating the dataset
        })
        return parent


class XYSeparateFileDataset(AbstractXYDataset):
    """Build dataset for examples that x and y are in different files."""

    def _build_dataset(self, files):
        if isinstance(files, list):
            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.flat_map(lambda f: tf.data.TextLineDataset(f).skip(self.config['skip_count']))
        else:
            dataset = tf.data.TextLineDataset(files)
        # repeat the dataset
        if self.config['repeat']:
            dataset = dataset.repeat(self.config['repeat'])
        return dataset

    def _build_dataset_from_files(self, files):
        """Build dataset from file(s).

        Args:
            files: An iterable of tuple (x_file, y_file)

        Returns:
            A tf.data.Dataset object
        """
        x_files, y_files = files

        x_dataset = self._build_dataset(x_files)
        y_dataset = self._build_dataset(y_files)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        return dataset

    def _build_dataset_from_files_for_predict(self, files):
        """Build dataset from file(s) for prediction.

        Args:
            files: An iterable of x_file

        Returns:
            A tf.data.Dataset object
        """
        x_files = files
        dataset = self._build_dataset(x_files)
        return dataset

    def _get_default_config(self):
        parent = super(XYSeparateFileDataset, self)._get_default_config()
        parent.update({
            'skip_count': 0,  # may useful when need to skip head line in files
            'repeat': 1,  # repeating the dataset
        })
        return parent
