import csv
import os

import numpy as np
import skimage
import skimage.data
import skimage.transform
from sklearn.utils import shuffle


class DataIter(object):
    """Data iterator base class.
    Reference designs:
    https://github.com/dmlc/mxnet/blob/master/python/mxnet/io.py
    https://github.com/NervanaSystems/neon/blob/master/neon/data/dataiterator.py
    http://stackoverflow.com/questions/19151/build-a-basic-python-iterator
    """

    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        """Returns next data bath. Python 3 compatible."""
        return self.next()

    def reset(self):
        """Resets iterator."""
        raise NotImplementedError

    def next(self):
        """Returns next data batch.
        Child classes need to implement this function. Python 2 compatible.
        """
        raise NotImplementedError


class NDArrayIter(DataIter):
    def __init__(self,
                 data,
                 labels,
                 batch_size=1,
                 to_one_hot=False,
                 num_classes=None,
                 loop_over_batch=False,
                 shuffle_data=True,
                 data_dtype=np.float32,
                 labels_dtype=np.int32):
        """Numpy ndarray iterator.

        Args:
            data: Assume NHWC layout, for other layouts, assume N (num_samples)
                  is in the first dimension.
            labels: If already one-hot encoded, then (num_samples, num_classes);
                    otherwise should be (num_samples, ) with integer class label
                    with 0-index. Only expect 1d or 2d labels.
            batch_size: Batch size. If num_samples % batch_size != 0, then
                        the last batch will be smaller. If batch_size == 0 or 
                        -1, then use all data (no batch).
            to_one_hot: Convert to one_hot if original dataset is not one-hot
                        encoded.
            num_classes: Number of classes, needed if to_one_hot is true.
            loop_over_batch: Loop-over last batch to contain data from beginning
                             to make sure all batch will have size batch_size.
            data_dtype: Returned datase dtype.
            labels_dtype: Returned labels dtype.

        TODO: Support loop-over batch to solve last batch smaller problem.
        """
        super(NDArrayIter, self).__init__()

        # Sanity check
        if len(data) == 0:
            raise ValueError("Error: len(data) == 0.")
        if len(data) != len(labels):
            raise ValueError("Error: len(data) != len(labels).")
        if loop_over_batch:
            raise NotImplementedError("Loop over batch not implemented.")

        # Set dataset info
        self.num_samples = len(data)
        self.num_classes = num_classes

        # Data
        self.data = np.array(data).astype(data_dtype)

        # Labels
        ndim_labels = np.ndim(labels)
        if to_one_hot:
            # Convert labels to one-hot if necessary
            if num_classes is None:
                raise ValueError("When to_one_hot=True, num_classes can not be"
                                 "None.")
            if ndim_labels == 1 or ndim_labels == 2 and labels.shape[1] == 1:
                # Convert to one-hot
                self.labels = np.zeros((self.num_samples, self.num_classes))
                self.labels[np.arange(self.num_samples), labels] = 1
            elif ndim_labels == 2:
                # Already one-hot, do a sanity check
                if not labels.shape[1] == self.num_classes:
                    raise ValueError("labels.shape[1]: {}, shall equal to "
                                     "num_classes: {}".format(
                        labels.shape[1], self.num_classes))
                self.labels = labels
            else:
                raise ValueError("ndim(labels): {} > 3".format(np.ndim(labels)))
        else:
            # Do not perform any operation on labels
            self.labels = labels
        self.labels = np.array(self.labels).astype(labels_dtype)

        # Set parameters
        self.to_one_hot = to_one_hot
        self.loop_over_batch = loop_over_batch
        self.shuffle_data = shuffle_data
        if batch_size == 0 or batch_size == -1:
            self.batch_size = self.num_samples
        else:
            self.batch_size = batch_size

        # Set states
        self.start_idx = 0
        self.reset()

    def next(self):
        if self.start_idx >= self.num_samples:
            raise StopIteration
        else:
            end_idx = self.start_idx + self.batch_size
            batch_data = self.data[self.start_idx:end_idx]
            batch_labels = self.labels[self.start_idx:end_idx]
            self.start_idx = end_idx
            return (batch_data, batch_labels)

    def reset(self):
        if self.shuffle_data:
            self.data, self.labels = shuffle(self.data, self.labels)
        self.start_idx = 0


class MnistDataIter(NDArrayIter):
    def __init__(self,
                 root_dir,
                 dataset,
                 batch_size=1,
                 to_one_hot=True,
                 loop_over_batch=False,
                 shuffle_data=True,
                 data_dtype=np.float32,
                 labels_dtype=np.int32):
        """Mnist Data Iterator

        Example usage:
            mnist = MnistDataIter(data_root, 'train', batch_size=128)
            data, labels = mnist.next()
        """

        def load_train(root_dir):
            with open(os.path.join(root_dir, 'train-images-idx3-ubyte')) as f:
                data = np.fromfile(file=f, dtype=np.uint8)
                data = data[16:].reshape((60000, 28, 28, 1))
            with open(os.path.join(root_dir, 'train-labels-idx1-ubyte')) as f:
                labels = np.fromfile(file=f, dtype=np.uint8)
                labels = labels[8:].reshape((60000))
            return data, labels

        def load_test(root_dir):
            with open(os.path.join(root_dir, 't10k-images-idx3-ubyte')) as f:
                data = np.fromfile(file=f, dtype=np.uint8)
                data = data[16:].reshape((10000, 28, 28, 1))
            with open(os.path.join(root_dir, 't10k-labels-idx1-ubyte')) as f:
                labels = np.fromfile(file=f, dtype=np.uint8)
                labels = labels[8:].reshape((10000))
            return data, labels

        # Read from mnist file
        if dataset == 'train':
            data, labels = load_train(root_dir)
        elif dataset == 'test':
            data, labels = load_test(root_dir)
        elif dataset == 'all':
            train_data, train_labels = load_train(root_dir)
            test_data, test_labels = load_test(root_dir)
            data = np.concatenate((train_data, test_data), axis=0)
            labels = np.concatenate((train_labels, test_labels), axis=0)
        else:
            raise ValueError("dataset must be 'train' or 'test'")

        # Scale 255. to 1.
        data = data / 255.

        super(MnistDataIter, self).__init__(data,
                                            labels,
                                            batch_size=batch_size,
                                            to_one_hot=to_one_hot,
                                            num_classes=10,
                                            loop_over_batch=loop_over_batch,
                                            shuffle_data=shuffle_data,
                                            data_dtype=data_dtype,
                                            labels_dtype=labels_dtype)


class FixedShapeImageIter(DataIter):
    def __init__(self, file_list, im_path, width, height, channel,
                 batch_size=1, to_one_hot=False,
                 loop_over_batch=False, shuffle_data=True,
                 data_dtype=np.float32, labels_dtype=np.int32):
        """Image ndarray iterator

        Args:
            file_list: Contains image_path,label, one sample for line. e.g.
                       image_file_foo.jpg, 1
                       image_file_bar.jpg, 2
                       ...
            im_path: Path containing image. e.g.
                     image_path/image_file_foo.jpg
            width: The width to resize image
            height: The heights to resize image
            channel: The input channel
            batch_size: Batch size. If num_samples % batch_size != 0, then
                        the last batch will be smaller. If batch_size == 0 or 
                        -1, then use all data (no batch).
            to_one_hot: Convert to one_hot if original dataset is not one-hot
                        encoded.
            loop_over_batch: Loop-over last batch to contain data from beginning
                             to make sure all batch will have size batch_size.
            data_dtype: Returned datase dtype.
            labels_dtype: Returned labels dtype.

        TODO: Currently just call NDArrayIter (storing all image in memory),
              needs to load image dynamically later.
        """
        super(FixedShapeImageIter, self).__init__()

        # Read from file
        self.xs = np.zeros((0, height, width, channel))
        self.ys = []
        with open(file_list, 'r') as f:
            reader = csv.reader(f)
            for im_name, im_class in reader:
                # Read image
                im_rgb = skimage.data.imread(os.path.join(im_path, im_name))

                # 4d -> 3d
                # TODO: fix hard-coded assert 3d and remove alpha channel
                assert channel == 3 == im_rgb.ndim
                if im_rgb.shape[2] == 4:
                    # remove alpha channel
                    im_rgb = im_rgb[:, :, :3]

                # Reshape
                im_rgb = skimage.transform.resize(im_rgb,
                                                  (height, width, channel),
                                                  preserve_range=True)
                im_rgb = im_rgb.astype(np.uint8)

                # Expand dim to NHWC
                im_rgb = np.expand_dims(im_rgb, 0)

                # Read label
                im_class = int(im_class)

                # Append to memory
                self.xs = np.concatenate((self.xs, im_rgb))
                self.ys.append(im_class)

        # Create NDArrayIter
        self.data_iter = NDArrayIter(self.xs, self.ys,
                                     batch_size=batch_size,
                                     to_one_hot=to_one_hot,
                                     loop_over_batch=loop_over_batch,
                                     shuffle_data=shuffle_data,
                                     data_dtype=data_dtype,
                                     labels_dtype=labels_dtype)

    def next(self):
        return self.data_iter.next()

    def reset(self):
        self.data_iter.reset()


if __name__ == '__main__':
    # Test NDArrayIter
    xs = np.random.rand(1000, 32, 32, 3)
    ys = np.random.randint(0, 10, 1000)
    batch_size = 128
    data_iter = NDArrayIter(xs, ys, batch_size=batch_size)
    for xs_batch, ys_batch in data_iter:
        print("xs shape %s dtype %s; ys shape %s dtype %s" %
              (xs_batch.shape, xs_batch.dtype, ys_batch.shape, ys_batch.dtype))

    xs = np.random.rand(1000, 784)
    ys = np.random.randint(0, 10, 1000)
    batch_size = 128
    data_iter = NDArrayIter(xs, ys, batch_size=batch_size)
    for xs_batch, ys_batch in data_iter:
        print("xs shape %s dtype %s; ys shape %s dtype %s" %
              (xs_batch.shape, xs_batch.dtype, ys_batch.shape, ys_batch.dtype))

    # file_list = '../extra-data/file_list.txt'
    # im_path = '../extra-data'
    # width = 32
    # height = 32
    # channel = 3
    # data_iter = FixedShapeImageIter(file_list, im_path, width, height,
    #                                 channel, batch_size=1)
    # for xs_batch, ys_batch in data_iter:
    #     print("xs shape %s dtype %s; ys shape %s dtype %s" %
    #           (xs_batch.shape, xs_batch.dtype, ys_batch.shape, ys_batch.dtype))

    # Test MnistDataIter
    data_root = os.path.join(os.path.expanduser('~'), 'data/mnist')
    mnist = MnistDataIter(data_root, 'train', batch_size=128)
    for xs_batch, ys_batch in mnist:
        print("xs shape %s dtype %s; ys shape %s dtype %s" %
              (xs_batch.shape, xs_batch.dtype, ys_batch.shape, ys_batch.dtype))
