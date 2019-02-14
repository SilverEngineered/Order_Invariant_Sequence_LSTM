import tensorflow as tf
import numpy as np
import nnUtils
import data_utilsWDB as utils
import os
import dataset as data
from matplotlib import pyplot as plt
# HyperParameters
seq_batch_size = 3
num_epochs = 100
num_cycles=2000000
learning_rate = 0.0015
stride = 1

# Parameters
layer1_output_nodes = 32
layer2_output_nodes = 64
dense_layer1_output_nodes = 64
dense_layer2_output_nodes = 128
img_width = 28
img_height = 28
img_color_dim = 1
seq_length=10
T_steps=5
lamda=.001
mem_size=64
save_rate=5000

X = tf.placeholder(tf.float32, [None, img_width, img_height, img_color_dim])
X_seq = tf.placeholder(tf.float32, [None])
X_Train, Y_Train=data.import_mnist()
Qt=tf.placeholder(tf.float32, [None])
Ct=tf.placeholder(tf.float32, [None])

logs_path = 'tensorboard'
writer = tf.summary.FileWriter(logs_path)
dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint = os.path.join(dir_path,"checkpoints","mnist.ckpt")

#Embedding Network
def conv_net(X, mem_size,reuse=True):
    with tf.variable_scope("Convnet", reuse=tf.AUTO_REUSE):
        layer1 = nnUtils.create_new_conv_layer(X, layer1_output_nodes, [5, 5], [1, 1], stride, "layer1")
        layer2 = nnUtils.create_new_conv_layer(layer1, layer2_output_nodes, [5, 5], [1, 1], stride, "layer2")
        flattened = tf.contrib.layers.flatten(layer2)
        dense_layer1 = tf.layers.dense(flattened,128, activation=tf.nn.relu)
        dense_layer2=tf.layers.dense(dense_layer1,256, activation=tf.nn.relu)
        mem_vector = tf.layers.dense(dense_layer2, mem_size)
    return mem_vector

#Takes in q_t pointer and returns the new memory cell
def q_t_LSTM(q_t_p,ct,reuse=False):
    with tf.variable_scope("Memory/Cell", reuse=reuse):
        ht, ct =nnUtils.lstm(q_t_p,ct,mem_size*2)
        qt=tf.layers.dense(ht,mem_size)
        return qt, ct

#Achieves order invariance by using addition which in communicative
def process_block(T_steps,X,X_seq,m, q_t_p,ct, first_call=False):
    for t in range(T_steps):
        if first_call:
            q_t, ct =q_t_LSTM(q_t_p, ct,reuse=(t!=0))
        else:
            q_t, ct =q_t_LSTM(q_t_p, ct, reuse=True)
        r_t=[]
        e_j=[]
        x=[]
        qtt=tf.transpose(q_t)
        e_j=tf.matmul(m,qtt)
        a_j=tf.nn.softmax(e_j)
        for i in range(seq_length):
            r_t.append(a_j[i]*m[i])
        r_t=tf.reduce_sum(r_t,0)
        for i in range(mem_size):
            x.append(q_t[0][i])
        for i in range(mem_size):
            x.append(r_t[i])
        q_t_p=x
    return tf.concat([q_t[0],r_t],0), ct


#Performs the calculation for the guesses on how many of each digit there are
def write_block(qtp, reuse=True):
    with tf.variable_scope("Decoder",reuse=reuse):
        x=tf.transpose(tf.expand_dims(tf.expand_dims(tf.expand_dims(qtp, 1), 1),1))
        layer1 = nnUtils.create_new_conv_layer(x, layer1_output_nodes, [5, 5], [1, 1], stride, "layer1")
        flattened = tf.contrib.layers.flatten(layer1)
        dense_layer1 = tf.layers.dense(flattened,64, activation=tf.nn.relu)
        dense_layer2 = tf.layers.dense(dense_layer1,seq_length)
        mem_vector=tf.reshape(dense_layer2,[-1,seq_length])
        return mem_vector
def get_points(out):
    points=np.zeros(10)
    out_manip=[]
    outr=np.round(out)
    while(sum(outr)!=10):
        if(sum(outr)>10):
            big_d=np.argmax(outr-out)
            outr[big_d]-=1
        if(sum(outr)<10):
            big_d=np.argmax(out-outr)
            outr[big_d]+=1
    return outr
with tf.name_scope("ReadNetwork"):
    m=conv_net(X,mem_size,reuse=True)
    tensum=tf.summary.tensor_summary("Memory",m)

Process_init=process_block(T_steps,X,X_seq,m,Qt,Ct, first_call=True)
Process=process_block(T_steps,X,X_seq,m,Qt,Ct)
Write_init=write_block(Process_init[0], reuse=False)
Write=write_block(Process[0])
vars = tf.trainable_variables()
with tf.name_scope("Loss"):
    #write_loss=tf.reduce_mean(tf.nn.l2_loss(Write[0]-X_seq))
    write_loss=tf.reduce_mean(tf.losses.absolute_difference(Write[0],X_seq))
    tf.summary.scalar("Loss",write_loss)
#my_vars=[v for v in vars if (v.name.startswith("Decoder")out.clip(0, 10))]
my_vars=[v for v in vars if (v.name.startswith("Decoder") or v.name.startswith("Memory/Cell")or v.name.startswith("Convnet"))]
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(write_loss, var_list=my_vars)
iteration = tf.Variable(0, dtype=tf.int32)
init=tf.global_variables_initializer()
increment_iter = tf.assign(iteration, iteration + 1)
saver = tf.train.Saver()
full_path = tf.train.latest_checkpoint("checkpoints")
print("full_path = %s" % full_path)
with tf.Session() as sess:
    try:
        saver.restore(sess, full_path)
        sess.run(increment_iter)
        print("Loaded Checkpoint!")
    except:
        sess.run(init)
    writer.add_graph(sess.graph)
    total_loss=0
    recent_loss=0
    summ=tf.summary.merge_all()
    qt=[]
    ct=[]
    out_2=[]
    for i in range(num_cycles):
        xseqbatch,xbatch=data.generate_variable_sequence(X_Train,Y_Train,seq_length)
        if i==0:
            q_t_p=np.random.rand(mem_size*2).astype(np.float32)
            ct=np.random.rand(mem_size*2).astype(np.float32)
            qt, ct= sess.run(Process_init,{X: xbatch, X_seq: xseqbatch, Qt: q_t_p, Ct: ct})
            qt=np.transpose(qt)
            ct=np.transpose(ct[0])
            mem=sess.run(m,{X: xbatch})
            out=sess.run(Write_init,{m: mem, Qt: qt, Ct: ct})
        else:
            qt, ct= sess.run(Process,{X: xbatch, X_seq: xseqbatch, Qt: qt, Ct: ct})
            qt=np.transpose(qt)
            ct=np.transpose(ct[0])
            mem=sess.run(m,{X: xbatch})
            out=sess.run(Write,{m: mem, Qt: qt, Ct: ct})
        out_2 = point_alloc=get_points(out[0])
        #out_2=np.reshape(out_2,[1,-1])
        _, loss = sess.run([optimizer, write_loss], {
                       Write: out, X_seq: xseqbatch, X: xbatch, Qt: qt, Ct: ct})
        total_loss+=loss
        recent_loss+=loss
        accuracy=0
        for j in range(len(xseqbatch)):
            if out_2[j]<xseqbatch[j]:
                accuracy+=out_2[j]
            else:
                accuracy+=xseqbatch[j]
        accuracy=accuracy*100/seq_length
        if i%100==0:
           #summarytensor=sess.run(tensum, {Write: out, X_seq: xseqbatch, X: xbatch})
           summary=sess.run(summ,{Write: out, X_seq: xseqbatch, X: xbatch})
           writer.add_summary(summary,i)
           #writer.add_summary(summarytensor,i)
        if(i%500==0 and i>0):
            print(qt)
            #point_alloc=get_points(out[0])
            print("AVG Loss: " + str(total_loss/i))
            print("AVG Loss Last 500: " + str(recent_loss/500))
            print()
            print("Iteration: " + str(i))
            print("Loss: " + str(loss))
            print("Out: " + str(out[0]))
            print("Sum out: " + str(sum(out[0])))
            print("Points:      " + str(np.array(point_alloc)))
            print("Real Values: " + str(xseqbatch))
            print("% Correct: " + str(accuracy))
            print()
            recent_loss=0
        if i%save_rate==0:
            saver.save(sess, checkpoint, global_step=save_rate)
#sess.run(increment_iter)
