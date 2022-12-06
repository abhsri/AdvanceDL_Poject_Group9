import pandas as pd
import seaborn as sns
import matplotlib as plt


class DataRecorder():
    """
    Class for keeping track of loss and accuracy during training.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.reset(df)

    def reset(self, df: pd.DataFrame = None):
        """Function for reseting recorder."""
        if df is not None:
            self.record = df
        else:
            self.record = pd.DataFrame({
                'set': pd.Series(dtype='str'),
                'epoch': pd.Series(dtype='int'),
                'metric': pd.Series(dtype='str'),
                'value': pd.Series(dtype='float')
            })

    def record_accuracy(self, set: str, epoch: int, accuracy: float):
        """Records new datapoint for accuracy."""
        self.record = self.record.append({
            'set': set,
            'epoch': epoch,
            'metric': 'accuracy',
            'value': accuracy
        }, ignore_index=True)

    def record_loss(self, set: str, epoch: int, loss: float):
        """Records new datapoint for loss."""
        self.record = self.record.append({
            'set': set,
            'epoch': epoch,
            'metric': 'loss',
            'value': loss
        }, ignore_index=True)

    def info(self):
        return self.record.info()

    def head(self, n: int = 10):
        return self.record.head(n)

    def relplot(self):
        """Plot the loss and accuracy side by side."""
        sns.relplot(data=self.record, x='epoch', y='value', hue='set',
                    style='set', col='metric', kind='line')
        plt.show()

    def plot(self):
        """Plot the loss and accuracy side by side."""
        _, axs = plt.subplots(1, 2, figsize=(13, 6))
        sns.lineplot(
            data=self.record.query("metric == 'loss'"),
            x="epoch", y="value",
            hue="set", style="set", ax=axs[0]).set(title='Loss')
        sns.lineplot(
            data=self.record.query("metric == 'accuracy'"),
            x="epoch", y="value",
            hue="set", style="set", ax=axs[1]).set(title='Accuracy')
        plt.show()

