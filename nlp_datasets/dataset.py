import abc


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
