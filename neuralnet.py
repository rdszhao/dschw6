import numpy as np
import util
from copy import copy
from tqdm import tqdm

class Activation():
    def __init__(self, activation_type='relu'):
        self.activation_type = activation_type
        self.x = None

    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        if self.activation_type.lower() == "relu":
            return self.ReLU(z)
        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        if self.activation_type.lower() == "relu":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)

    def ReLU(self, x):
        return x * (x > 0)

    def output(self, x):
            x2 = x - (np.max(x) + np.min(x)) / 2
            x2 = np.clip(x2, -300, 300)
            e = np.exp(x2)
            esum = e.sum(axis=1, keepdims=True)
            return e / esum

    def grad_ReLU(self, x):
        return (x >= 0)

    def grad_output(self, x):
        return 1

class Layer():
    def __init__(self, in_units, out_units, activation=Activation('relu')):
        np.random.seed(42)
        self.w = 0.01 * np.random.random((in_units + 1, out_units))
        self.velocity = np.zeros((in_units + 1, out_units))
        self.x = None
        self.a = None
        self.z = None
        self.activation = activation  # Activation function
        self.dw = 0

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        X = util.append_bias(self.x)
        self.a = X @ self.w
        self.z = self.activation(self.a)
        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma, regularization, gradReqd=True):
        if self.activation.activation_type == "output":
            delta_j = deltaCur
        else:
            delta_j = self.activation.backward(self.a) * deltaCur
        X = util.append_bias(self.x)
        self.dw = X.T @ delta_j
        new_deltaCur = (delta_j @ self.w.T)[:, 1:]

        if gradReqd:
            regularization_type, gamma = regularization
            if regularization_type == "L2":
                weight_decay = gamma * self.w
            else:
                weight_decay = 0
            self.velocity = (momentum_gamma * self.velocity) + \
                learning_rate * (self.dw - weight_decay)
            self.w += self.velocity

        return new_deltaCur


class NeuralNetwork():
    def __init__(self, config=None):
        config = util.load_config('config.yaml')
        self.config = config
        self.stepsize = config['learning_rate']
        self.layers = []
        self.num_layers = len(self.config['layer_specs']) - 1
        self.x = None
        self.y = None
        self.targets = None
        self.config['layer_specs'] = np.array(self.config['layer_specs']).astype(int)

        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(
                        self.config['layer_specs'][i],
                        self.config['layer_specs'][i + 1]
                    )
                )
            elif i == self.num_layers - 1:
                self.layers.append(
                    Layer(
                        self.config['layer_specs'][i],
                        self.config['layer_specs'][i + 1],
                        Activation('output')
                    )
                )
        print('hi im alive')

    def __call__(self, x, targets=None):
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        self.x = np.copy(x)
        for layer in self.layers:
            x = layer.forward(x)
        self.y = x

        if targets is not None:
            self.targets = targets
            return np.mean(self.loss(x, targets)), util.calculateCorrect(x, targets)
        else:
            return self.y

    def predict(self, x):
        self.x = np.copy(x)
        for layer in self.layers:
            x = layer.forward(x)
        self.y = x
        return (self.y == self.y.max(axis=1)[:, None]).astype(int).T[0]

    def loss(self, logits, targets):
        t = targets
        y = np.clip(logits, 1e-40, 1)
        return -np.sum(t * np.log(y), axis=1)

    def backward(self, gradReqd=True):
        deltaCur = self.targets - self.y
        for layer in reversed(self.layers):
            deltaCur = layer.backward(
                deltaCur=deltaCur,
                learning_rate=self.stepsize,
                momentum_gamma=self.config['momentum_gamma'],
                regularization=(self.config['regularization_type'],
                                self.config['regularization_gamma']),
                gradReqd=gradReqd
            )

    def full_prop(self, x, targets=None, gradReqd=True):
        loss, accuracy = self.forward(x, targets)
        self.backward(gradReqd=gradReqd)
        return loss, accuracy

    def fit(self, x_train, y_train):
        new_model = copy(self)
        config = new_model.config
        training_stats = [new_model.full_prop(x_train, y_train) for epoch in tqdm(range(config['epochs']))]
        self.__dict__ = new_model.__dict__