import time
from typing import Dict
import tensorflow as tf
import sklearn as skl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from metric import metric
from .logging import DataRecorder
from .data import rand_augment


class UnderSupervisedLearner():
    def __init__(self, model, optimizer, loss_object_train, loss_object_test,
                 recorder: DataRecorder = None):
        # Set optimizer and loss function
        self.model = model
        self.optimizer = optimizer
        self.loss_object_train = loss_object_train
        self.loss_object_test = loss_object_test
        self.recorder = recorder

        # Keeps track of training loss and accuracy
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        # Keeps track of test loss and accuracy
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

        # Validation trackig
        self.valid_loss = []
        self.valid_nmi = []
        self.valid_ami = []
        self.valid_ari = []
        self.valid_acc = []

    def reset(self):
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

    @tf.function
    def train_step(self, a1, a2):

        with tf.GradientTape() as tape:
            s1 = self.model(a1, training=True)
            s2 = self.model(a2, training=True)
            loss = self.loss_object_train(s1, s2)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def valid_step(self, a1, a2):
        s1 = self.model(a1)
        s2 = self.model(a2)
        loss = self.loss_object_train(s1, s2)
        
        p1 = self.model.to_prediction(s1)
        p2 = self.model.to_prediction(s2)
        return loss, p1, p2

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object_test(labels, predictions)

        self.test_loss(t_loss)
        self.test_acc(labels, predictions)

    def fit(self, config: Dict, train_ds, test_ds, plot_every=0,
            early_stopping=False, patience=3, wait_epoch=1):
        """Training function to fit the network"""
        all_time = time.time()
        saved_loss = None
        saved_acc = None
        count = patience

        for epoch in range(config['EPOCHS']):
            epoch_time = time.time()
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.test_acc.reset_states()
            self.valid_loss = []
            self.valid_nmi = []
            self.valid_ami = []
            self.valid_ari = []
            self.valid_acc = []

            print(30*"=", "EPOCH", (epoch+1), 30*"=")

            # Training loop
            for _, *augments in tqdm(train_ds, "Training"):
                a1, a2 = rand_augment(augments, 2)
                loss = self.train_step(a1, a2)
                self.train_loss(loss)

            # Validation loop
            for _, *augments, test_labels in tqdm(test_ds, "Validation"):
                a1, a2 = rand_augment(augments, 2)
                loss, p1, p2 = self.valid_step(a1, a2)
                self.valid_loss.append(loss.numpy())
                #nmi, ami, ari, acc = metric(p1, p2)
                self.valid_nmi.append(skl.metrics.normalized_mutual_info_score(p1, p2))
                self.valid_ami.append(skl.metrics.adjusted_mutual_info_score(p1, p2))
                self.valid_ari.append(skl.metrics.adjusted_rand_score(p1, p2))
                self.valid_acc.append(skl.metrics.accuracy_score(p1, p2))


            try:
                epoch_lr = self.optimizer._decayed_lr(tf.float32).numpy()
            except:
                epoch_lr = self.optimizer.lr.numpy()
            # Print information about epoch
            print(f'Epoch {epoch + 1}, '
                  f'Learning Rate: {epoch_lr:.5f}, '
                  f'Epoch time: {time.time()-epoch_time:.2f} seconds')
                  
            print(f'Loss: {self.train_loss.result():.3f}, '
                  f'Valid Loss: {np.mean(self.valid_loss):.4f}, '
                  f'NMI: {np.mean(self.valid_nmi):.4f}, '
                  f'AMI: {np.mean(self.valid_ami):.4f}, '
                  f'ARI: {np.mean(self.valid_ari):.4f}, '
                  f'ACC: {np.mean(self.valid_acc):.4f}, '
                  )

            # record data
            if (self.recorder is not None):
                self.recorder.record_loss(
                    'train', (epoch+1), self.train_loss.result().numpy())
                self.recorder.record_loss(
                    'validation', (epoch+1), np.mean(self.valid_loss))
                self.recorder.record_accuracy(
                    'validation', (epoch+1), np.mean(self.valid_acc))
            if plot_every != 0 and (epoch + 1) % plot_every == 0:
                # Plot resutls so far
                self.recorder.plot()

            if early_stopping:
                if epoch > wait_epoch:
                    if saved_loss is None:
                        saved_loss = np.mean(self.valid_loss)
                        saved_acc = np.mean(self.valid_acc)
                        count = patience
                    elif np.mean(self.valid_loss) < saved_loss:
                        saved_loss = np.mean(self.valid_loss)
                        saved_acc = np.mean(self.valid_acc)
                        count = patience
                    elif count > 0:
                        count -= 1
                    elif count == 0:
                        print(f'Out of patience! Returning {saved_acc:.4f}')
                        return saved_loss, saved_acc

        print("")
        print(26*"<",
              f'FINISHED {(time.time()-all_time):.2f}s',
              26*">")

        return np.mean(self.valid_loss), np.mean(self.valid_acc)


def check_prediction(model, x, y, aug=None, rows=2, plot=True):
    # Get random samples from test set
    samples = np.random.randint(x.shape[0], size=rows*3)
    test_image, test_label = x[samples], y[samples]
    # Performe augmentation
    if aug is not None:
        aug_image = aug(test_image)
    else:
        aug_image = test_image
    # Predict class of image
    prediction = model.prediction(aug_image)
    if plot:
        plt.figure(figsize=(13, 5*rows))
        for i in range(aug_image.shape[0]):
            plt.subplot(rows, 3, (i+1))
            imgplot = plt.imshow(aug_image[i, :, :, 0])
            plt.title(f"GT={test_label[i]}, Prediction={prediction[i]}")


def check_augmentation(x, aug, rows=2):
    # Get random samples from test set
    samples = np.random.randint(x.shape[0], size=rows*3)
    test_image = x[samples]
    aug_test = aug(test_image)

    plt.figure(figsize=(13, 5*rows))
    for i in range(aug_test.shape[0]):
        plt.subplot(rows, 3, (i+1))
        imgplot = plt.imshow(aug_test[i, :, :, 0])

