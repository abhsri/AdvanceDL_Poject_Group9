import time
from typing import Dict
import tensorflow as tf
import sklearn as skl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .metric import metric, class_histogram
from .logging import DataRecorder
from .data import rand_augment


class UnderSupervisedLearner():
    def __init__(self, model, config, optimizer, loss_object_train,
                 recorder: DataRecorder=None):
        # Set optimizer and loss function
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.loss_object_train = loss_object_train
        self.recorder = recorder
        
        # Keeps track of training loss and accuracy 
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

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

        
    def fit(self, train_ds, test_ds, plot_every=0, early_stopping=False,
            patience=3, wait_epoch=1):
        """Training function to fit the network"""
        all_time = time.time()
        saved_loss = None
        saved_acc = None
        count = patience
        
        for epoch in range(self.config['EPOCHS']):
            epoch_time = time.time()
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.valid_loss = []
            self.valid_nmi = []
            self.valid_ami = []
            self.valid_ari = []
            self.valid_acc = []
            self.predictions = np.array([])


            print(30*"=", "EPOCH",(epoch+1), 30*"=")

            # Training loop
            for image, *augments in tqdm(train_ds, "Training"):
                a1, a2 = rand_augment(augments, 2)
                loss = self.train_step(a1, a2)
                self.train_loss(loss)

            # Validation loop
            for image, *augments, test_labels in tqdm(test_ds, "Validation"):
                a1, a2 = rand_augment(augments, 2)
                loss, p1, p2 = self.valid_step(a1, a2)
                self.predictions = np.concatenate((self.predictions, p1.numpy()))
                self.predictions = np.concatenate((self.predictions, p2.numpy()))

                self.valid_loss.append(loss.numpy())
                test_labels = test_labels.numpy()
                nmi, ami, ari, acc = metric(test_labels.reshape(-1),p1)
                self.valid_nmi.append(nmi)
                self.valid_ami.append(ami)
                self.valid_ari.append(ari)
                self.valid_acc.append(acc)


            epoch_lr = self.optimizer._decayed_lr(tf.float32).numpy()
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
                    'train', (epoch+1),self.train_loss.result().numpy())
                self.recorder.record_loss(
                    'validation', (epoch+1), np.mean(self.valid_loss))
                self.recorder.record_accuracy(
                    'validation', (epoch+1), np.mean(self.valid_acc))
            if plot_every != 0 and (epoch + 1) % plot_every == 0:
                # Plot resutls so far
                self.recorder.plot()
                cls_cnt, cls_list, cls_hist = class_histogram(
                    self.predictions, addon_title="Validation")
                comb = [(x, y) for x, y in zip(cls_list, cls_hist)]
                print("Class list: ", comb)

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
                elif count  == 0:
                    print(f'Out of patience! Returning {saved_acc:.4f}')
                    return saved_loss, saved_acc
                
        print("")
        print(26*"<", 
              f'FINISHED {(time.time()-all_time):.2f}s', 
              26*">") 
        
        return np.mean(self.valid_loss), np.mean(self.valid_nmi), np.mean(self.valid_ami), np.mean(self.valid_ari), np.mean(self.valid_acc)


def check_images(images, augment, size=(3,3)):
    
    cnt = size[0]*size[1]
    rows = size[0]
    columns = size[1]
    samples = np.random.randint(images.shape[0], size=columns)
    
    plt.figure(figsize=(5*rows, 10*columns))
    for r in range(rows):
        for c in range(columns):
            
            plt.subplot(rows, columns, (r*columns+c+1))
            if r == 0:
                imgplot = plt.imshow(images[samples[c]])
            else:
                aug_img = augment(images[samples[c]])
                imgplot = plt.imshow(aug_img)