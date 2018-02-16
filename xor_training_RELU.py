import numpy as np
from numpy import random as rd
import SimpleNN as snn
import SimpleNN.NetVisuals as nv


trainingset = [(0,0,0),(1,0,1),(0,1,1),(1,1,0)]

ina = snn.InputSource(0.0)
inb = snn.InputSource(0.0)
bs = snn.LayerBias()

hl = []
for n in range(0,16):
    hl.append(snn.Neuron(snn.act_RELU_safe,.1))
    hl[-1].add_child(ina,rd.rand())
    hl[-1].add_child(inb,rd.rand())
    hl[-1].add_child(bs,rd.rand())

on = snn.Neuron(snn.act_Sigmoid,.1)
for cn in hl:
    on.add_child(cn,rd.rand()/10-0.5)

out = snn.OutputSink(on)


vis = nv.NNPlotter([out],[-10,10])


tidx = 0
for v in range (0,50000):
    if (v%100) == 0:
        print(v)
        vis.update_plot()
    
    v1 = trainingset[tidx][0]
    v2 = trainingset[tidx][1]
    tgt = trainingset[tidx][2]

    ina.set_value(v1)
    inb.set_value(v2)

    val = out.compute_output()
    out.train_children(tgt)

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


vis.persist_plot()