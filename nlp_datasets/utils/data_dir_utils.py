import os


def get_data_dir():
    root = os.path.join(os.path.dirname(__file__), '../../')
    data_dir = os.path.join(root, 'data')
    return os.path.abspath(data_dir)


def get_data_file(filename):
    return os.path.abspath(os.path.join(get_data_dir(), filename))
