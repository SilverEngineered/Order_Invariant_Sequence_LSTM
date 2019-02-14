import tensorflow as tf
import numpy as np
import random as rand
def import_mnist(path="",batch_size=64):
    X_Train = 'train-images-idx3-ubyte.gz'
    Y_Train = 'train-labels-idx1-ubyte.gz'
    X_Test = 't10k-images-idx3-ubyte.gz'
    Y_Test = 't10k-labels-idx1-ubyte.gz'
    with open(path + X_Train, 'rb') as fp:
        X_Train = tf.contrib.learn.datasets.mnist.extract_images(fp)
    with open(path + Y_Train, 'rb') as fp:
        Y_Train = tf.contrib.learn.datasets.mnist.extract_labels(fp)
    print("Loaded data...")
    #X_Train=scale_data(X_Train)
    #dataset = tf.data.Dataset.from_tensor_slices((X_Train,Y_Train))
    #dataset=dataset.repeat()
    #dataset = dataset.shuffle(buffer_size=100)
    #dataset = dataset.batch(batch_size)
    #iter = dataset.make_one_shot_iterator()
    #el = iter.get_next()
    return X_Train,Y_Train
def scale_data(data, scale=[0, 1], dtype=np.float32):
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)
#Returns sequence vector, correct count vector, img array of this information
def generate_variable_sequence(X_Train,Y_Train,num_seq=10,max_seq_size=10):
    seq_vec_arr=[]
    count_vec_arr=[]
    img_array=[]
    #this_seq_size=rand.randint(0,max_seq_size)
    this_seq_size=num_seq
    seq_vec=[]
    count_vec=[0,0,0,0,0,0,0,0,0,0]
    img=[]
    for j in range(this_seq_size):
        data_index=rand.randint(0,X_Train.shape[0]-1)
        seq_vec.append(Y_Train[data_index])
        img.append(X_Train[data_index])
        count_vec[Y_Train[data_index]]=count_vec[Y_Train[data_index]]+1
    seq_vec_arr.append(seq_vec)
    count_vec_arr.append(count_vec)
    return count_vec,img
