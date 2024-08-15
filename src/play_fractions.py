from decimal import Decimal, getcontext
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from multiprocessing import Pool, cpu_count  

getcontext().prec = 128

def convert_to_decimal(arr):
    return np.array([Decimal(float(x)) for x in arr])

def parallel_convert_to_decimal(data):
    with Pool(cpu_count()) as pool:
        return np.array(pool.map(convert_to_decimal, data))

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            weight = parallel_convert_to_decimal(weight)
            bias = np.random.randn(layer_sizes[i + 1])
            bias = convert_to_decimal(bias)
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, x):
        return np.array([Decimal(1) / (Decimal(1) + Decimal(-val).exp()) for val in x])

    def sigmoid_derivative(self, x):
        return x * (Decimal(1) - x)

    def forward(self, inputs):
        self.activations = [inputs]
        for i in range(self.num_layers - 1):
            net_input = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            activation = self.sigmoid(net_input)
            self.activations.append(activation)
        return self.activations[-1]

    def backward(self, inputs, expected_output, learning_rate):
        output_errors = expected_output - self.activations[-1]
        deltas = [output_errors * self.sigmoid_derivative(self.activations[-1])]

        for i in range(self.num_layers - 2, 0, -1):
            error = np.dot(deltas[-1], self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            activation_reshaped = self.activations[i].reshape(-1, 1)
            delta_reshaped = deltas[i].reshape(1, -1)
            self.weights[i] += learning_rate * np.dot(activation_reshaped, delta_reshaped)

    def train(self, inputs, expected_output, test_inputs, test_outputs, learning_rate, epochs, batch_size):
        for epoch in range(epochs):
            test_accuracy = self.evaluate(test_inputs, test_outputs)

            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            inputs = inputs[indices]
            expected_output = expected_output[indices]

            progress_bar = tqdm(range(0, len(inputs), batch_size), desc=f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.2f}%")
            for start_idx in progress_bar:
                end_idx = min(start_idx + batch_size, len(inputs))
                batch_inputs = inputs[start_idx:end_idx]
                batch_outputs = expected_output[start_idx:end_idx]

                for input_vector, output_vector in zip(batch_inputs, batch_outputs):
                    self.forward(input_vector)
                    self.backward(input_vector, output_vector, learning_rate)
                
                loss = self.calculate_loss(batch_outputs)
                
                progress_bar.set_postfix(loss=loss)

    def calculate_loss(self, output_vectors):
        return np.mean([(self.activations[-1] - output_vector) ** 2 for output_vector in output_vectors])

    def evaluate(self, test_inputs, test_outputs):
        print("evaluat start")
        correct = 0
        total = len(test_inputs)
        for input_vector, output_vector in zip(test_inputs, test_outputs):
            output = self.forward(input_vector)
            if np.argmax(output) == np.argmax(output_vector):
                correct += 1
        print("evaluat end")
        return (correct / total) * 100


def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    print("data processed")

    # Sample 1/5 of the data
    train_indices = np.random.choice(len(train_dataset), len(train_dataset) // 5, replace=False)
    test_indices = np.random.choice(len(test_dataset), len(test_dataset) // 5, replace=False)

    x_train = train_dataset.data.numpy()[train_indices].reshape(-1, 28*28).astype(np.float32) / 255.0
    y_train = F.one_hot(train_dataset.targets[train_indices].clone().detach(), num_classes=10).numpy()
    x_test = test_dataset.data.numpy()[test_indices].reshape(-1, 28*28).astype(np.float32) / 255.0
    y_test = F.one_hot(test_dataset.targets[test_indices].clone().detach(), num_classes=10).numpy()
    print(y_test.shape)

    print("converting to decimal")
    x_train = parallel_convert_to_decimal(x_train)
    x_test = parallel_convert_to_decimal(x_test)
    y_train = parallel_convert_to_decimal(y_train)
    y_test = parallel_convert_to_decimal(y_test)

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":

    print("initializing nn")
    layer_sizes = [28*28, 128, 64, 32, 16, 16, 32, 64, 128, 10]
    nn = NeuralNetwork(layer_sizes)
    print("initialized nn")

    print("loading data")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print("loaded data")

    print("training nn")
    nn.train(x_train, y_train, x_test, y_test, learning_rate=Decimal(0.1), epochs=10, batch_size=32)
    print("trained nn")


    correct = 0
    for input_vector, output_vector in zip(x_test, y_test):
        prediction = nn.forward(input_vector)
        if np.argmax(prediction) == np.argmax(output_vector):
            correct += 1
    accuracy = correct / len(x_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    #convなしの場合は8epochで79.4%だった.