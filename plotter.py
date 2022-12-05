import numpy as np
from sklearn.manifold import TSNE
class label_plotter():
    def __init__(self, train_X, train_y, test_X, test_y, trasformed_train_X=None, trasformed_test_X=None):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.trasformed_X = trasformed_train_X
        self.trasformed_test_X = trasformed_test_X
        if self.trasformed_test_X is None:
            _, self.trasformed_test_X = self.dim_reduction(test_X)
        if self.trasformed_X is None:
            _, self.trasformed_X = self.dim_reduction(train_X)


    def plot(self, model, plot_data='test'):
        # plot can take 3 inputs 'test' , 'train' or 'all'
        if self.transform_obj is None:
            self.transform_obj = self.dim_reduction(train_X)
        
        plot_data = plot_data.lower()
        if (plot_data == 'train') or (plot_data == 'all'):
            pred_y = model.predict(self.train_X)
            self.plot_scatter(self.trasformed_X, pred_y)
        if (plot_data == 'test') or (plot_data == 'all'):
            pred_y = model.predict(self.test_X)
            self.plot_scatter(self.trasformed_test_X, pred_y)
    
    def plot_from_pred(self, pred_y, plot_data='train'):
        # plot can take 2 inputs 'test' or 'train'
        plot_data = plot_data.lower()
        if (plot_data == 'train'):
            self.plot_scatter(self.trasformed_X, pred_y)
        if (plot_data == 'test'):
            self.plot_scatter(self.trasformed_test_X, pred_y)
        

    def dim_reduction(self,X):
        # used data from training data
        transform_obj = TSNE()
        transformed_x = transform_obj.fit_transform(np.reshape(X, (X.shape[0],-1)))
        return transform_obj, transformed_x

    def plot_scatter(self, transformed_X, y):
        plt.scatter(transformed_X[:,0], transformed_X[:,1], c=y, alpha=0.2)
        #plt.show()

def main():
    #here is the usage
    plotter = label_plotter(train_X, train_y, test_X, test_y)
    plotter.plot_from_pred(test_y,'test' )