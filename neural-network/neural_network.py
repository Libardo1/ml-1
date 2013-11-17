import numpy as np

class NeuralNetwork:
    """Neural network using back-propagation algorithm"""

    layer_count = 0
    shape = None
    weights = []

    def __init__(self, layer_size):

        self.layer_count = len(layer_size) - 1
        self.shape = layer_size

        self._layer_input = []
        self._layer_output = []
        self._previous_weight_delta = []

        for (l1, l2) in zip(layer_size[:-1], layer_size[1:]):
            self.weights.append(np.random.normal(scale=0.1, size = (l2, l1+1)))
            self._previous_weight_delta.append(np.zeros((l2, l1+1)))

    def run(self, input):
        in_cases = input.shape[0]

        self._layer_input = []
        self._layer_output = []

        for index in range(self.layer_count):
            if index == 0:
                layer_input = self.weights[0].dot(np.vstack([input.T, np.ones([1, in_cases])]))
            else:
                layer_input = self.weights[index].dot(np.vstack([self._layer_output[-1],np.ones([1, in_cases])]))

            self._layer_input.append(layer_input)
            self._layer_output.append(self.sigmoid(layer_input))

        return self._layer_output[-1].T


    def sigmoid(self, x, derivative = False):
        if not derivative:
            return 1 / (1+np.exp(-x))
        else:
            out = self.sigmoid(x)
            return out * (1 - out)

    def train_epoch(self, input, target, training_rate = 0.2, momentum = 0.5):

        delta = []
        ln_cases = input.shape[0]

        self.run(input)

        for index in reversed(range(self.layer_count)):
            if index == self.layer_count - 1:
                output_delta = self._layer_output[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta *
                        self.sigmoid(self._layer_input[index], True))
            else:
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.sigmoid(self._layer_input[index], True)) 

        self.compute_weight_deltas(delta, input, ln_cases, training_rate,
                momentum)
        return error

    def compute_weight_deltas(self, delta, input, ln_cases, training_rate,
            momentum):
        for index in range(self.layer_count):
            delta_index = self.layer_count - 1 - index

            if index == 0:
                layer_output = np.vstack([input.T, np.ones([1, ln_cases])])
            else:
                layer_output = np.vstack([self._layer_output[index - 1], np.ones([1, self._layer_output[index - 1].shape[1]])])
            current_weight_delta = np.sum(layer_output[None,:,:].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0), axis = 0)
            weight_delta = training_rate * current_weight_delta + momentum * self._previous_weight_delta[index]
            self.weights[index] -= weight_delta

            self._previous_weight_delta[index] = weight_delta

if __name__ == "__main__":
    network = NeuralNetwork((2, 2, 1))
    print(network.weights)

    input = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    target_output = np.array([[0.05], [0.05], [0.95], [0.95]])

    iterations = 100000
    error = 1e-5
    for i in range(iterations + 1):
        err = network.train_epoch(input, target_output)
        if i % 2500 == 0:
            print("Iter {0}\t Error: {1:0.6f}".format(i, err))
        if err <= error:
            print("Min error reached at iter {0}".format(i))
            break
    output = network.run(input)

    print("input: {0}\nOutput: {1}".format(input, output))
