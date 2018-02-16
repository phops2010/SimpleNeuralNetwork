"""
Simple Neural Network Module that can be used to learn about how neural networks work. Please feel free
to modify anything you see, as this is a very simple module. Its not intended to be used for any hardcore deep learning,
and not robust enough to handle erroneous outputs. It is however fully functional and can be used to train neural networks.

Requires numpy. 


Uses the following objects that can be used to programmatically built neural networks:

 - InputSource: single input source to the neural network
 - LayerBias: sinlge input source to the neural network that allways has a value of 1.0. Used for biasing a neuron output
 - OutputSink: hooked to the output of a single neuron, used to compute the value of that neuron and train it
 - Neuron: main Neuron object

The module additional provides several activation function. The activation functions are in the form 

   act_Fun(val,dv=False)

where val is the value, and the dv is a flag that is set to False to evaluate the value of activation function, 
or set to true to evaluate the derivative of the activation function used for back propagation

The way to create a neural network using this module is:

1. Set up the InputSource objects and a single Bias ojbec
2. Create your hidden layer with an array of Neurons.
3. For each neuron of the hidden layer, use the add_Child function to add as the inputs and bias object 
   as children with inital weights
4. Repeat steps 2 and 3 for each layer, including the last layer
5. Create OutputSink objects for every output neuron that you wish to read output on. 

Computing the value of the network is done by calling compute_output on the OutputSink object
Training the network through 1 epoch with backpropagation is done by calling train_children(target) on every OutputSink object. 

Call the former 2 things in a loop, keeping track of the error to get the network to converge on a target

Author: Nikolai Vozza, 02/2017

"""



import numpy as np


class InputSource(object):
    """
    Input source for the network
    """

    def __init__(self,value):
        """
        Create an Input source with inital value
            value: intial value for the input source
        """
        self.value=value

    def set_value(self,value):
        """
        Set the value of the input source
            value: value for the input source

        """
        self.value = value

    def compute_output(self):
        """
        Compute output for the Input source. As this is the last link in the chain of computing output
        at the output neurons, it returns the set value
       
            return: value set previouisly
        """
        return self.value

    #training endpoint
    def train_and_propagate(self,de_do):
        """
        Since this is the training endpoint, does nothing, just returns
        """
        return



class LayerBias(object):
    """
    Bias source for the network. You can also use a Input Source with a value of 1.0, but this helps distinguish
    the bias from the real inputs
    """
    def __init__(self):
        """
        Does nothing
        """
        pass
    def compute_output(self):
        """
        Always returns 1.0

        return: 1.0
        """
        return 1.0

    def train_and_propagate(self,de_do):
        """
        Since this is the training endpoint, does nothing, just returns
        """
        return

#output sink
class OutputSink(object):
    """
    Endpoint in the neural network. Used to compute outputs and train the network
    """
    def __init__(self, source_neuron):
        """
        Create an output sink

            source_neuron: single neuron on the output layer
        """
        self.source_neuron= source_neuron
        self.target = 0.0
        self.state = []
 
    #computes the main output
    def compute_output(self):
        """
        Computes the output of the neuron it was hooked up to

            return: output of the neuron. Returns a copy of a numpy object, so you can work with it without issues. 
        """
        self.state = self.source_neuron.compute_output()
        return np.copy(self.state)

    #trains the network
    def train_children(self,target):
        """
        Trains the output neuron with backpropagation for a given target value 

            target: target value to train.  
        """
        self.target = target
        if not self.state:
            self.state = self.source_neuron.compute_output()
        de_do = -(target-self.state) #compute the derivative of error with respect to output
        self.source_neuron.train_and_propagate(de_do) #start the backpropagation
    


class Neuron(object):
    """
    Main Neuron class.
    """
    def __init__(self, activation_func, learning_rate=.01, name="neuron"):
        """
        Create a new neuron

            activation_func: activation function. Use your own in the format supplied in the SimpleNN help doc, or use one of the provided ones
            learning_rate: learning rate for the neuron
            name: individual name for the neuron

        """
        self.learning_rate = learning_rate
        self.af = activation_func
        self.name = name

        self.weights = np.zeros(0)
        self.child_neurons = []
        self.state = 0.0
        self.inputs = np.zeros(0)
    
    def add_child(self, child_neuron, weight):
        """
        Add a child to the neuron. Every childs output is taken into the neuron and multiplied by the provided weight. 

            child_neuron: Neuron, InputSource, or LayerBias object
            weight: inital weight of the child output

        """
        self.child_neurons.append(child_neuron)
        self.weights = np.append(self.weights,weight)
        self.inputs = np.append(self.inputs,0.0)

    def compute_output(self):
        """
        Computes the output of the neuron

        return: output of the neuron

        """

        #store the inputs and the sum of the weighted inputs, called state here,  for use  back propagation
        self.state = 0.0
        for idx,tn in enumerate(self.child_neurons):
            self.inputs[idx] = tn.compute_output()
        self.state = np.dot(self.weights,self.inputs) 
        to = self.af(self.state,dv=False)
        return to

    def train_and_propagate(self,de_do):
        """
        Adjust weights based on error

            de_do: Change in error (de) with respect to change in the output (d0)

        """
        #compute the doutput/dsuminputs. This is the derivative of the activation function. 
        do_ds = self.af(self.state,dv=True)
        
        #train myself
        for k in range(0,len(self.weights)):
            de_dw = de_do * do_ds * self.inputs[k] # de/dweight = de/doutput * doutput/dsuminputs * dsuminputs/dw
            self.weights[k] = self.weights[k] - self.learning_rate * de_dw 
       
        #propagate the error to hidden layers
        for idx,tn in enumerate(self.child_neurons):
            de_do_prop = de_do * do_ds * self.weights[idx] # de/do_hidden = de/dout * dout/dsuminputs * dsuminptus/doutputhidden
            tn.train_and_propagate(de_do_prop)
        
def act_RELU(val,dv=False):
    """
    Rectified Linear Unit activation. y=x for x>0, y=0 for x<0

    specify dv to true to compute the derivative of the function
    """
    if dv:
        if val>0.0:
            return 1
        else:
            return 0
    else:
        if val>0.0:
            return(val)
        else:
            return 0

def act_RELU_safe(val,dv=False):
    """
    Rectified Linear Unit activation with safe negative space for recovery

    specify dv to true to compute the derivative of the function
    """
    if dv:
        if val>0.0:
            return 1
        else:
            return .01
    else:
        if val>0.0:
            return(val)
        else:
            return 0.01*val


def act_Linear(val, dv=False):
    """
    Linear Unit activation. y=x. Used for function approximation. 

    specify dv to true to compute the derivative of the function
    """
    if dv:
        return 1
    else:
        return val

def act_Sigmoid(val, dv=False):
    """
    Your run of the mill sigmoid. 

    specify dv to true to compute the derivative of the function
    """
    if dv:
        return act_Sigmoid(val,dv=False)*(1-act_Sigmoid(val,dv=False))
    else:
        return (1.0)/(1.0+np.exp(-val))


        

    
        

    
    


