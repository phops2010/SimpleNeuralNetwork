import numpy as np
from numpy import random as rd
import time
import SimpleNN as snn

import matplotlib.pyplot as plt

#define a target funciton
def target_function(val):
    return (1/50.0)*((0.75) * (val**3) - (2.4) * (val**2) - (5.15) * (val) + 41)

#set up the data for plotting the actual function
px = np.linspace(0,5,20)
py = target_function(px)

#build the network with 1 input, 1 output and 10 hidden units
bias=snn.LayerBias()
input1 = snn.InputSource(0)

hidden_layer = []
for k in range(0,10):
    hidden_layer.append(snn.Neuron(snn.act_Sigmoid,.1))
    hidden_layer[-1].add_child(input1,rd.rand()*10-5)
    hidden_layer[-1].add_child(bias,rd.rand()*10-5)

on=snn.Neuron(snn.act_Linear,.1)
for cn in hidden_layer:
    on.add_child(cn,rd.rand()*5.0-2.5)
    
Out = snn.OutputSink(on)

#plot the inital estimate
ey = np.zeros(len(py))
for idx,x in enumerate(px):
    input1.set_value(x)
    ey[idx] = Out.compute_output()

fig = plt.figure()
plt.plot(px,py)
plt.plot(px,ey)
plt.show(block=False)


#iterate a bunch
for v in range(0,100000):

    #update the plot for every 1000 iterations
    if (v%1000)==0:
        for idx,x in enumerate(px):
            input1.set_value(x)
            ey[idx] = Out.compute_output()
        plt.clf()
        plt.plot(px,py)
        plt.plot(px,ey)
        print(v)
        fig.canvas.draw()      
   
    #pick a random domain point to simulate a data point
    ival = rd.rand()*5.0

    #compute the actual target value
    tval = target_function(ival)

    #compute the output
    input1.set_value(ival)
    Out.compute_output()

    #train the network
    Out.train_children(tval)
 

#show the final plot
for idx,x in enumerate(px):
    input1.set_value(x)
    ey[idx] = Out.compute_output()

fig = plt.figure()
plt.plot(px,py)
plt.plot(px,ey)
plt.show()
   