from fc_layer import *
from generate_data import *
import numpy as np
import matplotlib.pyplot as plt

class ThreeLayerFC:
    def __init__(self,input_dim=2, hidden_dim1=10, hidden_dim2=10,num_classes=2,
                 weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim1)
        self.params['b1'] = np.zeros((1, hidden_dim1))
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim1, hidden_dim2)
        self.params['b2'] = np.zeros((1, hidden_dim2))
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim2, num_classes)
        self.params['b3'] = np.zeros((1, num_classes))


    def forward_backward(self,X,Y=None,Train=True):
        N = X.shape[0]
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        h1, cache1 = affine_relu_forward(X, W1, b1)
        h2, cache2 = affine_relu_forward(h1, W2, b2)
        out, cache3 = affine_forward(h2, W3, b3)
        scores = out  # (N,C)
        if Train == False:
            return scores
        loss, grads = 0, {}
        data_loss, dscores = softmax_loss(scores, Y)
        reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)+0.5 * self.reg * np.sum(W3 * W3)
        loss = data_loss + reg_loss

        dh2, dW3, db3 = affine_backward(dscores, cache3)
        dh1, dW2, db2 = affine_relu_backward(dh2, cache2)
        dX, dW1, db1 = affine_relu_backward(dh1, cache1)
        dW3 += self.reg * W3
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3
        return loss, grads

    def train(self,X,Y,learning_rate=1e-3,batch_size=10,epoch=5):
        num_train = X.shape[0]
        iter_step = int(num_train/batch_size)
        loss_history = []
        for epoch in range(epoch):
            for step in range(iter_step):
                sample_index = np.random.choice(num_train, batch_size, replace=True)
                X_batch = X[sample_index, :]
                Y_batch = Y[sample_index]
                loss, grads = self.forward_backward(X_batch, Y_batch)
                loss_history.append(loss)
                dW1 = grads['W1']
                dW2 = grads['W2']
                dW3 = grads['W3']
                db1 = grads['b1']
                db2 = grads['b2']
                db3 = grads['b3']

                self.params['W1'] -= learning_rate * dW1
                self.params['W2'] -= learning_rate * dW2
                self.params['W3'] -= learning_rate * dW3
                self.params['b1'] -= learning_rate * db1
                self.params['b2'] -= learning_rate * db2
                self.params['b3'] -= learning_rate * db3
        return loss_history

    def predict(self,X):
        hidden1 = np.maximum(0, np.dot(X, self.params['W1']) + self.params['b1'])
        hidden2 = np.maximum(0, np.dot(hidden1, self.params['W2']) + self.params['b2'])
        y_pred = np.argmax(np.dot(hidden2, self.params['W3']+self.params['b3']), axis=1)
        # scores = np.dot(hidden2, self.params['W3']+self.params['b3'])
        return y_pred


if __name__ == "__main__":
    X1, Y1 = generate_data()
    x_train, y_train, x_test, y_test = data_split(X1, Y1)
    # data = batch(x_train, y_train)
    model = ThreeLayerFC()
    loss_history=model.train(x_train,y_train,epoch=10)
    y_pred = model.predict(x_test)
    print("accuracy: %.2f " % (np.mean(y_pred == y_test) * 100),"%")
    plt.plot(loss_history)
    plt.show()