"""
Visual tools for plotting neural networks. Plots the map of  neurons(magenta), inputs(green), and biases(black),
and changes the color of the connecting lines based on weight values

Use help() for details


"""


import matplotlib.pyplot as plt
import numpy as np
import time
import copy

from . import Neuron,OutputSink,LayerBias,InputSource

weigth_cm = plt.get_cmap('seismic')

class NNPlotter(object):
    """
    Plotter class for visualising neural networks
    """
    
    def __init__(self,outputset,wrange=[-1,1]):
        """
        Creates a new plotter object from a given array of

            outputset: set of OutuputSinks from which to generate plot
            wrange: range of weight values for color scaling
        """
        self.wrange = wrange
        self.endpoints = []
        for cn in outputset:
            if isinstance(cn,OutputSink):
                self.endpoints.append(cn.source_neuron)
            elif isinstance(cn,Neuron):
                self.endpoints.append(cn)

        #compute deepest layer

        nlayers = 0
        for c in self.endpoints:
            tl = self._deepest_layer(c)
            if (tl>nlayers):
                nlayers = tl
        
        #build structure
        self.plt_objects = []
        thislayer= self.endpoints
        maxy = 0
        xval = 0
        while len(thislayer) != 0:
            lplot = {}
            nextlayer = []
            #compute the y spacing
            if len(thislayer)%2 == 0:
                yval = len(thislayer)/2.0-0.5
            else:
                yval = int(len(thislayer)/2.0)
            if yval>maxy:
                maxy = yval
            for tn in thislayer:
                #generate the circles
                if isinstance(tn,Neuron): 
                    for cn in tn.child_neurons:#build up the next layer from children
                        if cn not in nextlayer:
                            nextlayer.append(cn)
                    neuron_circ = plt.Circle((xval,yval),.2,color = 'm') 
                elif isinstance(tn,LayerBias):
                    neuron_circ = plt.Circle((xval,yval),.2,color = 'k') 
                else:
                    neuron_circ = plt.Circle((xval,yval),.2,color = 'g')
                lplot.update({tn:{'center':[xval,yval],'artist':neuron_circ,'w_lines':[]}})
                yval -= 1.0
                    

            self.plt_objects.append((lplot,xval))
            xval -= 5.0
            thislayer = nextlayer #move on to the next layer

        minx = xval

        #now, iterate again and add the weight lines
        
        for k in range(0,len(self.plt_objects)-1):
            (layer,xval) = self.plt_objects[k]
            (layer2,xval2) = self.plt_objects[k+1]
            for xn in layer.keys():
                if isinstance(xn,Neuron):
                    xset = [layer[xn]['center'][0],xval2]
                    yset= [layer[xn]['center'][1],0.0]
                    for cn in xn.child_neurons:
                        yset[1] = layer2[cn]['center'][1]
                        layer[xn]['w_lines'].append(plt.Line2D(np.array(xset),np.array(yset)))
                


        #finally show the plot
        self.fig = plt.figure()
        self.ax= self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
      
        for (layer,_) in self.plt_objects:
            for k in layer.keys():
                self.ax.add_artist(layer[k]['artist'])
                for l in layer[k]['w_lines']:
                    self.ax.add_line(l)
        self.ax.axis('equal')
        self.ax.axis([minx,1,-maxy-1,maxy+1])
        plt.show(block=False)
        plt.ion()
  
    def update_limits(self):
        """
        Updates the current weight range
          
        """
        minv = 0.0
        maxv = 0.0
        fset = False
        for (layer,_) in self.plt_objects:
            for k in layer.keys():
                if isinstance(k,Neuron):
                    for wval in k.weights:
                        if not fset:
                            minv = wval
                            maxv = wval
                            fset = True
                        else:
                            if (wval<minv):
                                minv = wval
                            if (wval>maxv):
                                maxv = wval
        self.wrange[0] = minv
        self.wrange[1] = maxv

    
    def update_plot(self):
        """
        Updates the plot
          
        """
        for (layer,_) in self.plt_objects:
            for k in layer.keys():
                if isinstance(k,Neuron):
                    for wval,lobj in zip(k.weights,layer[k]['w_lines']):
                        cind = (wval-self.wrange[0])/(self.wrange[1]-self.wrange[0])
                        cval = weigth_cm(cind)
                        lobj.set_color(copy.deepcopy(cval))
        self.fig.canvas.flush_events()
        time.sleep(.0001) #prevents the plot from freezing on lost focus 

    def _deepest_layer(self,c_neuron):
        """
        Computes the deepest layer of the network
          
        """
        cdepth = 0
        if isinstance(c_neuron, LayerBias) or isinstance(c_neuron, InputSource):
            return 1
        for cn in c_neuron.child_neurons:
            tdepth = self._deepest_layer(cn)
            if (tdepth>cdepth):
                cdepth=tdepth
        return 1 + cdepth

    def persist_plot(self):
        """
        Makes the plot block until closed
          
        """
        plt.figure(self.fig.number)
        plt.show(block=True)
