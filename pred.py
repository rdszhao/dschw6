# %%
import numpy as np
from neuralnet import NeuralNetwork
from util import load_data, normalize

# this was a model adapted from CSE151B: deep learning
# figured i wanted to try to adapt aspects of that including backprop
# in this hw even though backprop is out of this scope 
# but since its still pretty much from scratch i figured it might be valid
# it doesn't work how i want to and i think ive honestly
# kind of given up on fixing it. my mind is tired
# and if this thing refuses to cooperate then so be it

x_train, y_train = load_data('data/train-data.csv', norm=True)
x_test = normalize(np.loadtxt('data/test-data.csv', delimiter=','))
train_set = (x_train, y_train)

model = NeuralNetwork()
stats = model.fit(*train_set)
y = model.predict(x_train)

