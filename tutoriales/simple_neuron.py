import math

def sigmoid(x):
    y = 1.0 / (1.0 + math.exp(-x))
    return y

def actvate(inputs, weights):
    #perform net inputs
    h = 0
    for x, w in zip(inputs, weights):
        h += x*w

    #perfomr activation
    return sigmoid(h)


if __name__=="__main__":
    inputs = [0.5,0.3,0.2]
    weights = [0.4,0.7,0.2]
    output = actvate(inputs, weights)
    print(output)