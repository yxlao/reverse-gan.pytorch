import csv
import os
import numpy as np
from sklearn.utils import shuffle
import skimage
import skimage.data
import skimage.transform


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
    def __init__(self, data, labels, batch_size=1, to_one_hot=False,
                 loop_over_batch=False, shuffle_data=True,
                 data_dtype=np.float32, labels_dtype=np.int32):
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
            loop_over_batch: Loop-over last batch to contain data from beginning
                             to make sure all batch will have size batch_size.
            data_dtype: Returned datase dtype.
            labels_dtype: Returned labels dtype.
        TODO: Support loop-over batch to solve last batch smaller problem.
        TODO: Implement to_one_hot. Currently this can be covered from TF's
              tf.one_hot()
        """
        super(NDArrayIter, self).__init__()

        # Sanity check
        if len(data) == 0:
            raise ValueError("Error: len(data) == 0.")
        if len(data) != len(labels):
            raise ValueError("Error: len(data) != len(labels).")
        if to_one_hot:
            raise NotImplementedError("To one-hot not implemented.")
        elif np.array(labels).ndim != 1:
            raise ValueError("labels must be of shape (num_samples, ).")
        if loop_over_batch:
            raise NotImplementedError("Loop over batch not implemented.")

        # Set dataset info
        self.num_samples = len(data)

        # Set parameters
        self.data = np.array(data).astype(data_dtype)
        self.labels = np.array(labels).astype(labels_dtype)
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


class FixedShapeImageIter(DataIter):
    def __init__(self, file_list, im_path, width, height, channel,
                 batch_size=1, to_one_hot=False,
                 loop_over_batch=False, shuffle_data=True,
                 data_dtype=np.float32, labels_dtype=np.int32):
        """Image ndarray iterator.
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
                # read image
                im_rgb = skimage.data.imread(os.path.join(im_path, im_name))

                # 4d -> 3d
                # TODO: fix hard-coded assert 3d and remove alpha channel
                assert channel == 3 == im_rgb.ndim
                if im_rgb.shape[2] == 4:
                    # remove alpha channel
                    im_rgb = im_rgb[:, :, :3]

                # reshape
                im_rgb = skimage.transform.resize(im_rgb,
                                                  (height, width, channel),
                                                  preserve_range=True)
                im_rgb = im_rgb.astype(np.uint8)

                # expand dim to NHWC
                im_rgb = np.expand_dims(im_rgb, 0)

                # read label
                im_class = int(im_class)

                # append to memory
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
