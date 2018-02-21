import numpy as np
from numpy import random as rd
import SimpleNN as snn
import SimpleNN.NetVisuals as nv


trainingset = [(0,0,0),(1,0,1),(0,1,1),(1,1,0)]

ina = snn.InputSource(0.0)
inb = snn.InputSource(0.0)
bs = snn.LayerBias()

h1 = snn.Neuron(snn.act_Sigmoid,.5)
h1.add_child(ina,rd.rand()/10.0-.05)
h1.add_child(inb,rd.rand()/10.0-.05)
h1.add_child(bs,rd.rand()/10.0-.05)

h2 = snn.Neuron(snn.act_Sigmoid,.5)
h2.add_child(ina,rd.rand()/10.0-.05)
h2.add_child(inb,rd.rand()/10.0-.05)
h2.add_child(bs,rd.rand()/10.0-.05)

on = snn.Neuron(snn.act_Sigmoid,.5)
on.add_child(h1,rd.rand()/10.0-.05)
on.add_child(h2,rd.rand()/10.0-.05)
on.add_child(bs,rd.rand()/10.0-.05)

out = snn.OutputSink(on)

vis = nv.NNPlotter([out],[-10,10])


tidx = 0
for v in range (0,15000):
    if (v%100) == 0:
        print(v)
        vis.update_plot()
    
    v1 = trainingset[tidx][0]
    v2 = trainingset[tidx][1]
    tgt = trainingset[tidx][2]

    ina.set_value(v1)
    inb.set_value(v2)

    val = out.compute_output()
    out.train_children(-(tgt-val))

    tidx += 1
    if tidx > 3:
        tidx = 0


for tidx in range(0,4):
    v1 = trainingset[tidx][0]
    v2 = trainingset[tidx][1]
    ina.set_value(v1)
    inb.set_value(v2)
    val = float(out.compute_output())
    print("{:f},{:f} -> {:f}".format(v1,v2,val))


print(on.weights)
print(h1.weights)
print(h2.weights)
vis.persist_plot()