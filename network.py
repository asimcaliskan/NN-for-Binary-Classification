import numpy as np
def load_dataset(file_name):
    dataset_file = open(file_name, "r")
    data= []
    for line in dataset_file:
        data.append(([float(i) for i in line.split(",")[:-1]],int(line.split(",")[-1])))
    dataset_file.close()
    np.random.shuffle(data)
    return data

def separated_dataset(file_name, n_train_data):
    dataset = load_dataset(file_name)
    train_data = [(np.reshape(x,(1,4)), make_vector(y)) for x, y in dataset[0:n_train_data]]
    test_data  = [(np.reshape(x,(1,4)), make_vector(y)) for x, y in  dataset[n_train_data:]]
    return (train_data, test_data)

def make_vector(v):
    vector = np.zeros((1,1))
    vector[0] = v
    return vector

train_data, test_data = separated_dataset("C:\\Users\\MONSTER\\.spyder-py3\\CNN\\data_banknote_authentication.txt", 1000)


def sigmoid(z, derivative = False):
    if derivative == False:   
        return 1.0 /(1.0 + np.exp(-z))
    else:
        return sigmoid(z) * (1 - sigmoid(z))

def quadratic_cost(y, y_hat, derivative = False):
    if derivative == False:
        return (y - y_hat)**2 / 2
    else:
        return y_hat - y 

class FC:
    def __init__(self, in_size, out_size, activation):
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.W = np.random.randn(out_size, in_size)
        
    def forward(self, a_prev):
        self.a_prev = a_prev
        self.z = np.dot(self.W, self.a_prev)
        self.a = self.activation(self.z)
        return self.a
    
    # def backward(self, error, last_layer, learning_rate):
    #     if last_layer:
    #         print(self.activation(self.z, derivative=True).shape, error.shape,  self.a_prev.shape)
    #         delta = np.dot(error, self.activation(self.z, derivative=True).transpose())
    #         self.W = self.W - learning_rate * np.dot(delta, self.a_prev.transpose())
    #     else:
    #         print( error.shape, self.W.transpose().shape,  self.activation(self.z, derivative=True).shape, self.a_prev.shape)
    #         delta = np.dot(error, np.dot(self.W.transpose(), self.activation(self.z, derivative=True)))
    #         self.W = self.W - learning_rate * np.dot(delta, self.a_prev.transpose())
    #     return delta
    



num_epoch       = 10
mini_batch_size = 5
train_data_size = 1000
test_data_size  = 300
learning_rate   = 2
lambd           = 1
fc1 = FC(4, 2, sigmoid)
fc2 = FC(2, 1, sigmoid)

for epoch in range(num_epoch):
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    for data_index in range(0, train_data_size, mini_batch_size):
        d_loss_d_a3 = 0
        for mini_batch in range(mini_batch_size):
            y  = train_data[data_index + mini_batch][1]
            a1 = train_data[data_index + mini_batch][0]
            a2 = fc1.forward(a1.transpose())
            a3 = fc2.forward(a2)
            d_loss_d_a3 += quadratic_cost(y, a3, derivative=True)
        
        d_loss_d_a3 = d_loss_d_a3 / mini_batch_size
        d_z3_d_a2   = fc2.W.transpose()
        d_z2_d_a1   = fc1.W.transpose()
        d_a3_d_z3   = sigmoid(fc2.z, derivative=True)
        d_a2_d_z2   = sigmoid(fc1.z, derivative=True)
        
        
        error_2 = np.multiply(d_loss_d_a3, d_a3_d_z3)#................d_loss_d_z3
        error_1 = np.multiply(d_a2_d_z2, np.dot(d_z3_d_a2, error_2))#.d_loss_d_z2
        
        
        d_z3_d_w2   = fc2.a_prev.transpose()
        d_loss_d_w2 = np.dot(error_2, d_z3_d_w2)
        fc2.W       = np.subtract(fc2.W, learning_rate * d_loss_d_w2)  
        d_z2_d_w1   = fc1.a_prev.transpose()
        d_loss_d_w1 = np.dot(error_1, d_z2_d_w1)
        fc1.W       = np.subtract(fc1.W, learning_rate * d_loss_d_w1)     
        
    sigma = 0
    for data_index in range(test_data_size):
        y  = test_data[data_index][1]
        a1 = test_data[data_index][0]
        a2 = fc1.forward(a1.transpose())
        a3 = fc2.forward(a2)
        if np.round(a3)[0][0] == y[0][0]:
            sigma += 1

    print("Epoch = {} <-> Accuracy = {}".format(epoch, sigma / test_data_size))
