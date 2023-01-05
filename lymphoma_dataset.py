import tensorflow as tf
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


class LymphomaDataset:

    def __init__(self, dataset_path, preprocess_fn_kwargs, dataset_kwargs, train_val_test_split,
                 seed):
        """
        :param dataset_path: path to the dataset
        :param preprocess_fn_kwargs: arguments for the preprocessing function
        :param dataset_kwargs: arguments for the dataset caching and prefetching etc.
        :param train_val_test_split: list containing the split ratios for train, validation and test
        :param seed: seed for the random split
        """
        self.dataset_path = dataset_path
        self.preprocess_fn_kwargs = preprocess_fn_kwargs
        self.dataset_kwargs = dataset_kwargs
        assert len(train_val_test_split) == 3, "train_val_test_split must be a list of length 3"
        assert np.allclose(np.sum(train_val_test_split), 1), "train_val_test_split must sum to 1"
        self.train_val_test_split = train_val_test_split
        self.seed = seed

        train_split, val_split, test_split = self._split()

        self.train_dataset = self._create_dataset(train_split, 'train')
        self.val_dataset = self._create_dataset(val_split, 'val')
        self.test_dataset = self._create_dataset(test_split, 'test')

    def _split(self):
        """
        Splits the dataset into train, validation and test
        :return: train, validation and test splits
        """
        df = []  # store file names and labels
        for directory, subdirectories, files in os.walk(self.dataset_path):
            for file in files:
                # Get the full path of the file
                file_path = os.path.join(directory, file)

                # Get the label from the directory name
                label = os.path.basename(directory)

                # Append the data to the list
                df.append([file_path, label])

        # Create a DataFrame from the data
        df = pd.DataFrame(df, columns=['file_path', 'label'])
        # transform labels to integers
        df['label'] = pd.factorize(df['label'])[0].astype(np.uint8)
        # perform stratified train, validation and test split
        train_test_ss = StratifiedShuffleSplit(n_splits=1, test_size=self.train_val_test_split[2],
                                               random_state=self.seed)
        train_val_ss = StratifiedShuffleSplit(n_splits=1, test_size=self.train_val_test_split[1] / (
                1 - self.train_val_test_split[2]), random_state=self.seed)

        for train_index, test_index in train_test_ss.split(df, df['label']):
            train_val_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

        for train_index, val_index in train_val_ss.split(train_val_df, train_val_df['label']):
            train_df = train_val_df.iloc[train_index]
            val_df = train_val_df.iloc[val_index]

        return train_df, val_df, test_df

    def _create_dataset(self, df, split):
        """
        Creates a tf.data.Dataset from a pandas DataFrame
        :param df: pandas DataFrame containing the file paths and labels
        :param split: split name, one of 'train', 'val' or 'test'
        :return: tf.data.Dataset
        """
        # create dataset from pandas DataFrame containing file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((df['file_path'].values, df['label'].values))
        # load images
        load_func = lambda file_path, label: tf.numpy_function(self._load_image, [file_path, label],
                                                               [tf.float32, tf.uint8])
        dataset = dataset.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # preprocess images, create patches and return them with their labels
        dataset = dataset.map(self._preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # cache dataset, up to this point the transformations need to be applied only once
        if self.dataset_kwargs['cache_file'] is not None:
            if self.dataset_kwargs['cache_file'] == 'memory':
                dataset = dataset.cache()
            else:
                dataset = dataset.cache(self.dataset_kwargs['cache_file'] + '_{}'.format(split))

        # shuffle dataset if training
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=self.dataset_kwargs['buffer_size'], seed=self.seed)

        # repeat dataset indefinitely
        dataset = dataset.repeat()

        # apply data augmentation transformations if training (after caching)
        if split == 'train':
            dataset = dataset.map(self._augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # batch dataset,
        dataset = dataset.batch(self.dataset_kwargs['batch_size'],
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # reshape to 4D tensor with shape (batch_size*num_patches, patch_size, patch_size, channels)
        dataset = dataset.map(self._reshape_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # prefetch dataset
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    @staticmethod
    def _load_image(file_path, label):
        """
        Loads an image from a file path
        :param file_path: path to the image
        :param label: label of the image
        :return: image, label
        """
        # read image
        image = tf.keras.utils.load_img(file_path)  # load image in rgb format

        # convert to tensor
        image = tf.keras.utils.img_to_array(image, dtype=np.float32)

        return image, label

    def _preprocess_fn(self, image, label):
        """
        Preprocesses the image
        :param image: image
        :param label: label of the image
        :return: preprocessed image
        """

        # here apply color space conversion if specified, before patch extraction and normalization
        # TODO: add color space conversion

        # normalize image
        # TODO: check if different normalization is needed for different color spaces, standardize?
        image = image / 255.

        # extract patches
        image = tf.expand_dims(image, axis=0)
        patches = tf.image.extract_patches(images=image,
                                           sizes=self.preprocess_fn_kwargs['patch_sizes'],
                                           strides=self.preprocess_fn_kwargs['patch_strides'],
                                           rates=self.preprocess_fn_kwargs['patch_rates'],
                                           padding=self.preprocess_fn_kwargs['patch_padding'])

        # reshape patches
        patches = tf.reshape(patches, shape=[-1, self.preprocess_fn_kwargs['patch_sizes'][1],
                                             self.preprocess_fn_kwargs['patch_sizes'][2],
                                             tf.shape(image)[-1]])

        return patches, label

    def _augment_fn(self, patches, label):
        """
        Applies data augmentation transformations to the patches
        :param patches: patches to be augmented
        :param label: label of the patches
        :return: augmented patches, labels
        """

        # TODO: add data augmentation transformations
        return patches, label

    @staticmethod
    def _reshape_fn(batches, label):
        """
        Reshapes the batches to a 4D tensor with shape
        (batch_size*num_patches, patch_size, patch_size, channels)
        :param batches: patches to be reshaped
        :param label: label of the patches
        :return: reshaped batches, labels
        """

        # get the number of patches per image
        shape = tf.shape(batches)
        # reshape to 4D tensor with shape (batch_size*num_patches, patch_size, patch_size, channels)
        batches = tf.reshape(batches, shape=[shape[0] * shape[1], shape[2], shape[3], shape[4]])
        # repeat labels num_patches times
        label = tf.repeat(label, shape[1])

        return batches, label

    def get_datasets(self):
        """
        Returns the train, validation and test datasets
        :return: train, validation and test datasets
        """
        return self.train_dataset, self.val_dataset, self.test_dataset


if __name__ == '__main__':

    # Limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    dataset_path = r'C:\Users\loren\Datasets\lymphoma'
    preprocess_fn_kwargs = {'patch_sizes': [1, 128, 128, 1], 'patch_strides': [1, 64, 64, 1],
                            'patch_rates': [1, 1, 1, 1], 'patch_padding': 'VALID'}
    dataset_kwargs = {'batch_size': 4, 'cache_file': 'memory', 'buffer_size': 100}
    train_val_test_split = [0.6, 0.2, 0.2]
    seed = 42

    dataset = LymphomaDataset(dataset_path, preprocess_fn_kwargs, dataset_kwargs,
                              train_val_test_split, seed)

    train_dataset, val_dataset, test_dataset = dataset.get_datasets()

    # check if dataset is working
    for patches, labels in train_dataset.take(1):
        print(patches.shape, labels.shape)
        # plot some patches
        fig, axes = plt.subplots(3, 3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(patches[i*100, :, :, :])
            ax.set_title('Label: {}'.format(labels[i*100].numpy()))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
