import numpy as np
import scipy.special
import matplotlib.pyplot


class NeuralNetwork:

    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать количество узлов во входном, скрытом и выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Матрицы весовых коэффициентов связей wih и who.
        # Весовые коэффициенты связей между узлом i и узлом j
        # следующего слоя обозначены как w_i_j: w11, w21, w12 итд.
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.hnodes, self.inodes))

        # Коэффициент обучения
        self.lr = learningrate

        # Использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

        pass


    # Тренировка нейронной сети
    def train(self, input_list, targets_list):
        # targets_list = тренировочные примеры

        # Преобразовать список входных значений в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # Рассчитать входящие сигналы для выходного слоя 
        final_outputs = self.activation_function(hidden_inputs)

        # Рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # Рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # Ошибки выходного слоя = (целевое зн-е - фактическое зн-е)
        output_errors = targets - final_outputs

        # Ошибки скрытого слоя - это распределенные пропорционально весовым коэфф-там связей
        # и рекомбинированным на скрытых узлах
        hidden_errors = np.dot(self.who.T, output_errors)

        # Обновить весовые коэфф-ты для свзей между скрытым и выходным слоями
        self.who += self.lr * np.dpt((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass


    # Опрос нейронной сети
    def query(self, input_list):
        # Преобразовать список входных значений в двумерный массив
        input = np.array(input_list, ndim=2).T

        # Рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # Рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # Рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# Количество входных, скрытых и выходных узлов
input_nodes = 28 * 28
hidden_nodes = 100
output_nodes = 10

# Коэффициент обучения равен 0,3
learning_rate = 0.3

# Создать эксемпляр нейронной сети
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Загрузить тестовый набор данных MNIST
training_data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Тренировка нейронной сети
# Перебрать все записи в тренировочном наборе данных
for record in training_data_list:
    # получить список значений
    all_values.record.split(',')
    
    # Масштабировать и сместить входные значения
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # Создать целевые входные значения (все равны 0.01 за исключением желаемого маркерного, равного 0,99)
    targets = np.zeros(output_nodes) + 0.01

    # all_values[0] - целевое маркерное значение для данной записи
    targets[int(all_values[0])] = 0.99

    nn.train(inputs, targets)
    pass


