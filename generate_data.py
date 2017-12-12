import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
np.random.seed(3)
def generate_data():
    X1, Y1 = make_classification(n_samples=400, n_features=2, n_redundant=0,
                                 n_clusters_per_class=1, n_classes=2)
    return X1,Y1


def data_split(X1,Y1,ratio=80):
    train_num = int(X1.shape[0]*0.01*ratio)
    # index = np.random.choice(X1.shape[0], train_num, replace=True)
    # x_train = X1[index, :]
    # y_train = Y1[index]
    x_train = X1[:train_num, :]
    y_train = Y1[:train_num]
    x_test = X1[train_num:X1.shape[0], :]
    y_test = Y1[train_num:X1.shape[0]]
    return x_train,y_train,x_test,y_test

def batch(X1,Y1,batch_size = 10):
    batch_num = int(X1.shape[0]/batch_size)
    data=[]
    for i in range(batch_num):
        x_batch = X1[(batch_size*i):(batch_size*(i+1)),:]
        y_batch = Y1[(batch_size*i):(batch_size*(i+1))]
        batch = [x_batch,y_batch]
        data.append(batch)
    return data


if __name__ == "__main__":
    X1, Y1 = generate_data()
    print(X1.shape, Y1.shape)
    x_train, y_train, x_test, y_test = data_split(X1,Y1)
    print(x_train.shape, y_train.shape,x_test.shape,y_test.shape)
    data = batch(x_train, y_train)
    print(data[0][0],data[0][1])
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
    plt.show()
