from typing import Dict
import numpy as np
import tensorflow as tf


class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = self.lr_warmup_cosine_decay(
            global_step=step, total_steps=self.total_steps,
            warmup_steps=self.warmup_steps, start_lr=self.start_lr,
            target_lr=self.target_lr, hold=self.hold)
        return tf.where(
            step > self.total_steps, 0.0, lr, name="learning_rate")

    def lr_warmup_cosine_decay(self, global_step, warmup_steps, hold=0,
                               total_steps=0, start_lr=0.0, target_lr=1e-3):
        learning_rate = 0.5 * target_lr * (1 + tf.cos(tf.constant(np.pi) * float(
            global_step - warmup_steps - hold) / float(
                total_steps - warmup_steps - hold)))
        warmup_lr = target_lr * (global_step / warmup_steps)

        if hold > 0:
            learning_rate = tf.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)
        learning_rate = tf.where(global_step < warmup_steps,
                                 warmup_lr, learning_rate)
        return learning_rate


def lr_schedular(config: Dict, schedule: str = None, batch_count: int = 0):
    """Returns a schedular object depending on string input"""
    if schedule == "WarmUpCosineDecay":
        total_steps = batch_count*config['EPOCHS']
        warmup_steps = int(0.05*total_steps)

        return WarmUpCosineDecay(start_lr=0.0, target_lr=config['MAX_LR'],
                                 warmup_steps=warmup_steps,
                                 total_steps=total_steps,
                                 hold=warmup_steps)
    else:
        return
