# Importing required packages
import numpy as np  # Used to implementing N-D matrix & tensors (Pure Python just supports 1D-arrays(lists))
import pandas as pd  # Used as IO
import math  # Used for mathematical functions or parameters
import matplotlib.pyplot as plt  # Used for plotting
import xlsxwriter
from sklearn.metrics import mean_squared_error


# Defining sigmoid activation function
def sig(x):
    return 1 / (1 + math.e ** -x)

# Reading the dataset file
data = pd.read_excel('4N(5-8)(50f).xlsx', header=None)
data = np.asarray(data)  # Convert to 2-D array

# Splitting the dataset into inputs & targets
input_data = data[:, :50]
targets_data = data[:, :50]

input_size = np.shape(input_data)[1]  # Dimension of input array column
rows_data = np.shape(input_data)[0]  # Number of rows in dataset

# normilization

for i in range(input_size):
    mean = np.mean(input_data[:, i])
    var = np.var(input_data[:, i])
    for j in range(rows_data):
        input_data[j][i] = (input_data[j][i] - mean) / math.sqrt(var)


# Defining hyper-parameters
lr = 0.0002 # 0.0001
n1 = 40  # number of neurons (Hidden layer)
n2 = input_size  # number of neurons (Out-put layer)
epoch = 300
train_test_ratio = 0.8


# Splitting data into train & test
data_train_size = round(train_test_ratio * rows_data)
data_test_size = round((1 - train_test_ratio) * rows_data)

# Initializing weights
w1_init = np.random.uniform(-1, 1, (input_size, n1))
w2_init = np.random.uniform(-1, 1, (n1, n2))

w1 = w1_init
w2 = w2_init

# Defining mean squared error for train & test
mse_train_new = []
mse_test_new = []
predict_epoch = np.zeros(data_test_size)
output = np.zeros([rows_data, n1])
mse_train_epoch = 0
mse_test_epoch = 0

# Defining net
for i in range(epoch):  # Iteration over epochs    i: number of epoch

    # Train
    train_predicted = []
    test_predicted = []
    for j in range(data_train_size):  # Iteration over rows   j: number of row

        #  Layer 1 (Hidden layer)
        net1 = np.dot(input_data[j, :], w1)  # Net: Summation of inputs * weights
        o1 = sig(net1)

        #  Layer 2 (Out-put layer)
        net2 = np.dot(o1, w2)
        o2 = net2

        #  Calculating error of the current row
        y = o2
        err = y - targets_data[j]
        train_predicted.append(y)


        # Hidden layer 1
        # w1 = np.transpose(w2)
        err = np.reshape(err, (input_size, 1))
        grad_w1 = -1 * lr *np.dot(np.dot(err,np.transpose((input_data[j, :]).reshape(input_size,1))),np.dot(np.transpose(w2),np.diag(o1 * (1 - o1))))
        grad_w1 = np.reshape(grad_w1, (np.shape(w1_init)[0], np.shape(w1_init)[1]))
        w1 = w1 + grad_w1

        # Out-put layer

        grad_w2 = -1 * lr * o1.reshape(n1,1) * np.transpose(err.reshape(input_size,1))    #khati
        # grad_w2 = -1 * lr * np.dot(o1.reshape(n1, 1), np.transpose(np.dot(np.diag(o2 * (1 - o2)), err)))   #sigmoid
        grad_w2 = np.reshape(grad_w2, (np.shape(w2_init)[0], np.shape(w2_init)[1]))
        w2 = w2 + grad_w2

        # ٍ Error of the row is added to error of the epoch
        output[j] = o1

    mse_train_epoch=mean_squared_error(targets_data[:data_train_size],train_predicted)
    mse_train_new.append(mse_train_epoch)

    # Test
    err_test_epoch = 0
    for j in range(data_test_size):  # Iteration over rows   j: number of row

        #  Layer 1 (Hidden layer)
        net1 = np.dot(input_data[j + data_train_size, :], w1)  # Net: Summation of inputs * weights
        o1 = sig(net1)

        #  Layer 2 (Out-put layer)
        net2 = np.dot(o1, w2)
        o2 = net2

        #  Calculating error of the current row
        y = o2
        err = y - targets_data[j + data_train_size]
        test_predicted.append(y)

        # ٍ Error of the row is added to error of the epoch
        output[j + data_train_size] = o1

        if j % 100 == 0 and i % 50 == 0:
            plt.plot(y, 'b')
            plt.plot(targets_data[j + data_train_size], 'r')
            plt.show()

    mse_test_epoch = mean_squared_error(targets_data[data_train_size:], test_predicted)
    mse_test_new.append(mse_test_epoch)


    print('Epoch', i + 1, 'MSE train', mse_train_epoch, 'MSE Test', mse_test_epoch)
    # plt.plot(mse_train_new, 'b')
    # plt.plot(mse_test_new, 'r')
    # plt.show()


plt.plot(mse_train_new, 'b')
plt.plot(mse_test_new, 'r')
plt.show()

workbook = xlsxwriter.Workbook('E:\\D\\arshad\\terme4\\prj\\autoencoder\\w_real\\w_moreLayer\\w.xlsx')
workbook = xlsxwriter.Workbook('w.xlsx')
worksheetW = workbook.add_worksheet()

row = 0
col = 0
for i in(w1):
    for j in range(40):
        worksheetW.write(row, col+j,i[col+j])
    row+=1
workbook.close()

# workbook = xlsxwriter.Workbook('E:\\D\\arshad\\terme4\\prj\\autoencoder\\w_real\\w_moreLayer\\f.xlsx')
workbook = xlsxwriter.Workbook('f.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0
for i in(output):
    for j in range(40):
        worksheet.write(row, col+j,i[col+j])
    row+=1
workbook.close()
