import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        param input_size: integer, number of features of the input
        param hidden_size: integer, arbitrary number of parameters
        param output_size: integer, number of classes

        Define simple two layer neural network with relu activation function.

        You need to create weights and biases for both layers with the correct 
        shapes. Pass values to self.params dict for later use.
        '''
        self.params = {}
        '''
        START YOUR CODE HERE
        '''
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / output_size) 
        # self.params['b1'] = np.random.randn(hidden_size, 1) / np.sqrt(hidden_size)
        self.params['b1'] = np.random.randn(hidden_size) / np.sqrt(hidden_size)
        # self.params['b1'] = np.zeros((hidden_size, 1))
        # self.params['b2'] = np.random.randn(output_size, 1) / np.sqrt(output_size)
        self.params['b2'] = np.random.randn(output_size) / np.sqrt(output_size)
        # self.params['b2'] = np.zeros((output_size, 1))
        '''
        END YOUR CODE HERE
        '''


    def loss(self, X, y, reg=0.0):
        '''
        param X: numpy.array, input features
        param y: numpy.array, input labels
        param reg: float, regularization value


        Return:
        param loss: Define loss with data loss and regularization loss
        param grads: Gradients for weights and biases
        '''

        # Unpack weights and biases
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        loss = 0.0
        grads = {}
        
        N = X.shape[0]

        # print (f"X shape[0]: {X.shape[0]}")
        # print (f"X shape; {X.shape}")

        '''
        START YOUR CODE HERE
        '''
        # Compute the forward pass: using
        # ReLU activation for Hidden layer and 
        # SoftMax activation for Output layer
        scores = None
        
        # First layer pre-activation
        z = np.dot(X, W1) + b1  # (N, num_hidden)
        # First layer activation
        h = np.maximum(z, 0)    # ReLU
        # Second layer pre-activation
        scores = np.dot(h, W2) + b2
        
        # print(f"Scores are: {scores}")


        if y is None:
            return scores

        # data normalization with sklearn MinMaxScaler
        scaler = MinMaxScaler()
        # fit and transform in one step
        normalized_scores = scaler.fit_transform(scores)

        # Second layer activation:
        # compute SoftMax probabilities
        exp_scores = np.exp(normalized_scores)   # (N, C)
        # out = exp_scores / np.sum(exp_scores, axis=1).reshape(N, 1)
        out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
        # compute Cross-Entropy loss on output
        corect_logprobs = -np.log(out[np.arange(N), y])
        data_loss  = np.sum(corect_logprobs) / N
        reg_loss = 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
        loss = data_loss + reg_loss

        # print(f"Loss values are: {loss}")
        

        # Backward pass/Back propagation: compute gradients
        dout = np.copy(out)  # (N, C)
        dout[np.arange(N), y] -= 1
        dout /= N
        
        # compute gradient for parameters
        grads['W2'] = np.dot(h.T, dout)      # (H, C)
        grads['b2'] = np.sum(dout, axis=0)      # (C,)

        dh = np.dot(dout, W2.T)
        # Backprop the ReLU non-linearity
        # dh[h <= 0] = 0
        dz = dh * (z > 0)  # (N, H)

        grads['W1'] = np.dot(X.T, dz)        # (D, H)
        grads['b1'] = np.sum(dz, axis=0)       # (H,)
        
        # add reg term
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1

        # print(f"Grad values are: {grads}")
    
        '''
        END YOUR CODE HERE
        '''
        
        return loss, grads


    def train(self, X_train, y_train, X_val, y_val, learning_rate=1e-3, batch_size=4, num_iters=100):
        '''
        param X_train: numpy.array, trainset features 
        param y_train: numpy.array, trainset labels
        param X_val: numpy.array, valset features
        param y_val: numpy.array, valset labels
        param learning_rate: float, learning rate should be used to updated grads
        param batch_size: float, batch size is the number of images should be used in single iteration
        param num_iters: int, number of iterations you want to train your model

        method will return results and history of the model.
        '''

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        num_train = X_train.shape[0]

        for it in range(num_iters):
            # Create batches
            X_batch, y_batch = None, None
            '''
            START YOUR CODE HERE
            '''
            random_idxs = np.random.choice(num_train, batch_size)
            X_batch = X_train[random_idxs]
            y_batch = y_train[random_idxs]
            
            '''
            END YOUR CODE HERE
            '''
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch)

            # update weights and biases
            '''
            START YOUR CODE HERE
            '''
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']            
            '''
            END YOUR CODE HERE
            '''
            if (it+1) % 100 == 0:
                print(f'Iteration {it+1} / {num_iters} : {loss}')

            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()

            loss_history.append(loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

        return {'loss_history': loss_history, 'train_acc_history': train_acc_history, 'val_acc_history': val_acc_history}
    

    def predict(self, X):
        '''
        param X: numpy.array, input features matrix
        return y_pred: Predicted values

        Use trainied weights to do prediction for the given features 
        '''
        y_pred = None

        '''
        START YOUR CODE HERE
        '''
        params = self.params
        z = np.dot(X, params['W1']) + params['b1']
        h = np.maximum(z, 0)
        out = np.dot(h, params['W2']) + params['b2']
        y_pred = np.argmax(out, axis=1)
        '''
        END YOUR CODE HERE
        '''
        return y_pred