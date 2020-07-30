import numpy as np
def load_dataset(file_name):
    dataset_file = open(file_name, "r")
    data= []
    for line in dataset_file:
        data.append(([float(i) for i in line.split(",")[:-1]],int(line.split(",")[-1])))
    dataset_file.close()
    return data
def separated_dataset(file_name, n_train_data):
    dataset = load_dataset(file_name)
    train_data = [(np.reshape(x,(4,1)), make_vector(y)) for x, y in dataset[0:n_train_data]]
    test_data  = [(np.reshape(x,(4,1)), make_vector(y)) for x, y in  dataset[n_train_data:]]
    return (train_data, test_data)

def make_vector(v):
    vector = np.zeros((1,1))
    vector[0] = v
    return vector
#separated_dataset("C:\\Users\\MONSTER\\eclipse-workspace\\MyAI\\data_banknote_authentication.txt", 1000)

