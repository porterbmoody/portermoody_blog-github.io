import random
import math
import pandas as pd
import numpy as np
import time

break_ = '='*50

class NeuralNetwork():

    def __init__(self, inputs, outputs, learning_rate=.4) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.__weights = []
        self.__init_hidden_layer_weights__()
        self.__init_output_layer_weights__()
        self.__init_b__()
        self.__init_hidden__()
        self.__init_predicted_outputs__()

    def __init_hidden_layer_weights__(self):
        # matrix = np.array([[self.get_random_dec() for i in range(len(self.inputs))]])
        # for i in range(len(self.inputs) - 1):
            # matrix = np.append(matrix, [[self.get_random_dec() for i in range(len(self.inputs))]], axis = 0)
        matrix = np.array([
                [self.get_random_dec(), self.get_random_dec(), self.get_random_dec()],
                [self.get_random_dec(), self.get_random_dec(), self.get_random_dec()]
            ])
        self.__weights.append(matrix)

    def __init_output_layer_weights__(self):
        # matrix = np.array([[self.get_random_dec() for i in range(len(self.outputs))]])
        # for i in range(len(self.outputs) - 1):
            # matrix = np.append(matrix, [[self.get_random_dec() for i in range(len(self.outputs))]], axis = 0)
        matrix = np.array([
            [self.get_random_dec(), self.get_random_dec()]
        ])
        self.__weights.append(matrix)

    def __init_b__(self):
        self.__b = []
        for i in range(2):
            self.__b.append(self.get_random_dec())

    def __init_hidden__(self):
        self.__hidden = [0 for i in range(len(self.inputs))]

    def __init_predicted_outputs__(self):
        self.predicted_outputs = []
        for _ in range(len(self.outputs)):
            self.predicted_outputs.append([0 for i in range(len(self.outputs[0]))])

    def __compute_hidden_layer(self):
        self.__hidden = self.__weights[0].dot(self.inputs[self.current_index])
        for i, __hidden in enumerate(self.__hidden):
            self.__hidden[i] = self.sigmoid(__hidden + self.__b[0])

    def __compute_output_layer(self):
        self.predicted_outputs[self.current_index] = self.__weights[1].dot(self.__hidden)
        for i, __hidden in enumerate(self.__hidden):
            self.__hidden[i] = self.sigmoid(__hidden + self.__b[1])
        #     output_sum = self.__hidden.dot(self.__weights[1][i]) + self.__b[1]
        #     self.predicted_outputs[i] = self.sigmoid(output_sum)

    def __calculate_output_layer_gradient(self, i):
        epic1, epic2, epic3 = 0, 0, 0
        epic1 = round(-(self.outputs[self.current_index][i] - self.predicted_outputs[self.current_index][i]), 20)
        epic2 = round(self.predicted_outputs[self.current_index][i] * (1 - self.predicted_outputs[self.current_index][i]), 20)
        epic3 = round(self.__hidden[i], 20)
        return round(epic1 * epic2 * epic3, 20)

    def __calculate_hidden_layer_gradient(self, i):
        errors = []
        for i, output in enumerate(self.outputs[self.current_index]):
            E_o_out_o2   = round(-(self.outputs[self.current_index][i] - self.predicted_outputs[self.current_index][i]), 20)
            out_o_net_o2 = round(self.predicted_outputs[self.current_index][i] * (1 - self.predicted_outputs[self.current_index][i]), 20)
            net_o_out_h1 = round(self.__weights[1][i][0], 20)
            error = E_o_out_o2 * out_o_net_o2 * net_o_out_h1
            errors.append(error)

        E_total_out_h1 = sum(errors)
        out_h1_net_h1 = self.__hidden[0] * (1 - self.__hidden[0])
        net_h1_w1 = self.inputs[self.current_index][i]
        return E_total_out_h1 * out_h1_net_h1 * net_h1_w1

    def __update_hidden_layer_weights(self):
        for i, weight_array in enumerate(self.__weights[0]):
            gradient = self.__calculate_hidden_layer_gradient(i)
            for weight_index, weight in enumerate(self.__weights[0][i]):
                self.__weights[0][i][weight_index] = weight - (self.learning_rate * gradient)

    def __update_output_layer_weights(self):
        # there's a lot to iterate through here
        for i, weight_array in enumerate(self.__weights[1]):
            gradient = self.__calculate_output_layer_gradient(i)
            for weight_index, weight in enumerate(self.__weights[1][i]):
                self.__weights[1][i][weight_index] = weight - (self.learning_rate * gradient)

    def run_propogation(self, n):
        start = time.time()
        for _ in range(n):
            for i, input_vector in enumerate(self.inputs):
                self.current_index = i
                self.__compute_hidden_layer()
                self.__compute_output_layer()
                self.__update_hidden_layer_weights()
                self.__update_output_layer_weights()
        end = time.time()
        self.runtime = end - start

    def display_results(self):
        total_error = []
        for predicted_output, output in zip(self.predicted_outputs, self.outputs):
            total_error.append(sum([(1/2) * (predicted_output - actual_output)**2 for predicted_output, actual_output in zip(predicted_output, output)]))

        print("TOTAL ERROR:", sum(total_error))
        print("PREDICTED OUTPUTS:", self.predicted_outputs)
        print("RUNTIME:", round(self.runtime, 2))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))
    
    @staticmethod
    def get_random_dec():
        return random.randint(0, 10)/10
