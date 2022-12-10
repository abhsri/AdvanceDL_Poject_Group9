import os
import random
import tensorflow as tf
import imgaug.augmenters as iaa


class DataLoader():
    def __init__(self, in_path, absolute_in_path=False, mkdir=False):
        self.path = self._set_in_path(in_path, absolute_in_path, mkdir)
        self._load, self._save = self._set_tf_func()

    def _set_in_path(self, in_path, absolute_in_path, mkdir):
        if absolute_in_path:
            path = in_path
        else:
            path = os.path.join(os.getcwd(), in_path)

        # check if folder exists
        if os.path.exists(path) is False:
            if mkdir is False:
                raise NotADirectoryError(
                    f"Directory '{path}' does not exist and 'mkdir'=False")
            else:
                os.makedirs(path)
        return path

    def _set_tf_func(self):
        if "2.9" in tf.__version__:
            return tf.data.experimental.load, tf.data.experimental.save
        else:
            return tf.data.Dataset.load, tf.data.Dataset.save

    def load(self):
        return self._load(self.path)

    def save(self, dataset):
        return self._save(dataset, self.path)

    def exists(self):
        return os.path.exists(os.path.join(self.path, "dataset_spec.pb"))

    def get_train(self, config, pre_process=None):
        if self.exists():
            # Load existing dataset
            print("Stored training dataset found, loading...", end="")
            train_ds = self.load()
            print("[done]")
            print("Loaded dataset information:")
            for image, *augments in train_ds:
                print(f' => Batch Count: {len(train_ds)}')
                print(f' => Batch Size: {len(image)}')
                print(f' => Augmentations: {len(augments)}')
                break

        else:
            print("Generating new test augmentation dataset...")
            (x_train, y_train), (x_test,
                                 y_test) = config['DATASET'].load_data()

            if pre_process is not None:
                x_train = pre_process(x_train)

            # Generate augmentations
            aug_x_train = gen_augment(x_train, n_augments=config['N_AUG'])
            # Create training dataset
            train_ds = tf.data.Dataset.from_tensor_slices(
                (x_train, *aug_x_train)).shuffle(
                    x_train.shape[0]).batch(config['BATCH_SIZE'])
            # Save new augment
            print("Saving new train dataset...", end="")
            self.save(train_ds)
            print("[done]")

        assert isinstance(train_ds, tf.data.Dataset)
        return train_ds

    def get_test(self, config, pre_process=None):
        if self.exists():
            # Load existing dataset
            print("Stored test dataset found, loading...", end="")
            test_ds = self.load()
            print("[done]")
            print("Loaded dataset information:")
            for image, *augments, test_labels in test_ds:
                print(f' => Batch Count: {len(test_ds)}')
                print(f' => Batch Size: {len(image)}')
                print(f' => Augmentations: {len(augments)}')
                break

        else:
            # Generate augmentations
            print("Generating new test augmentation dataset...")
            (x_train, y_train), (x_test,
                                 y_test) = config['DATASET'].load_data()

            if pre_process is not None:
                x_train = pre_process(x_train)

            aug_x_test = gen_augment(x_test, n_augments=config['N_AUG'])
            # Create test dataset
            test_ds = tf.data.Dataset.from_tensor_slices(
                (x_test, *aug_x_test, y_test)).batch(config['BATCH_SIZE'])
            # Save new augment
            print("Saving new test dataset...", end="")
            self.save(test_ds)
            print("[done]")

        # Assert Dataset datatype
        assert isinstance(test_ds, tf.data.Dataset)
        return test_ds


def gen_augment(images, n_augments=1, normalize=None):
    """Generate n number of augmentations of each training image"""
    augments = []
    for i in range(n_augments):
        # Generate augmentation
        augment = get_aug_seq(images=images)

        # Normalize dataset
        if normalize is not None:
            augment = augment / normalize

        # Store augmentation
        augments.append(augment)
        print(f' => augment {i+1} complete')

    return augments


def rand_augment(aug_list, n_augments):
    """Selects a n random items from the list of augmentations"""
    return random.choices(aug_list, k=n_augments)


def get_aug_seq(images):
    """Generates a sequential function for data augmentation"""
    aug_seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),
                                  per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )], random_order=True)

    return aug_seq(images=images)
