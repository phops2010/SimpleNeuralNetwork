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
    Bias source for the network. You can also use a Input Source with a constant value, but this helps distinguish
    the bias from the real inputs
    """
    def __init__(self,value=1.0):
        """
        Initializes a bias. Default value is 1.0

            value: bias value
        """
        self.value = value

    def compute_output(self):
        """
        Returns the bias value

            return: bias value
        """
        return self.value

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
    def train_children(self,error_gradient):
        """
        Trains the output neuron with backpropagation for a given target value 

            error_gradient: error gradient to train on
        """
        self.gradient = error_gradient
        if not self.state:
            self.state = self.source_neuron.compute_output()
        self.source_neuron.train_and_propagate(self.gradient) #start the backpropagation
    
    def get_state(self):
        return np.copy(self.state)


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

class SoftMaxClassifier(object):
    """
    Softmax Classifier. This is made as a separate class because it varies in computation from traditional
    back propagation.

    It is composed of a single hidden layer of RELU units, feeding into a softmax classifier neuron for each group
    """
    def __init__(self,input_size,hidden_size,n_groups,reg=0.001,lr=1):
        """
        Initialize the classifer

            input_size: dimention of the input
            hidden_size: number of hidden neurons
            n_groups: number of output groups
            reg: regulation parameter
            lr: learning step size
        """
        self.s_in = input_size
        self.s_h = hidden_size
        self.s_g = n_groups

        #initalize the weights. rows are the input sources, columns are the index of the next neuron in line
        self.w_hidden = 0.01*np.random.randn(self.s_in,self.s_h)
        self.w_output = 0.01*np.random.randn(self.s_h,self.s_g)
        self.b_h = np.zeros((1,self.s_h)) # bias weights for hidden layer
        self.b_o = np.zeros((1,self.s_g)) # bias weights for output layer

        self.reg = reg
        self.step = lr

        
    
    def train_step(self,point_set,group_index):
        """
        Performs a single training step 

            point_set: input data set. Number of columns must match input_size parameter when the classifier is created
            group_index: vector of group membership for each point in range 0,n_roups. Must match the rows of point_set

        """
        npoints = point_set.shape[0]
        #compute the hidden layer output. 
        h_state = np.maximum(0,np.dot(point_set,self.w_hidden)+self.b_h)
        o_state = np.dot(h_state,self.w_output)+self.b_o

        #compute the output_probs. This is the softmax classifer
        numerator = np.exp(o_state)
        denomenator = np.sum(numerator, axis=1, keepdims=True) #sum over the output of the group classifiers
        probs = numerator/denomenator 
        
        #backprobagate the error
        dE = probs
        dE[range(npoints),group_index] -= 1.0
        dE = dE/npoints
        
        #adjust the weights of the output
        d_w_output = self.step*(np.dot(h_state.T,dE) + (self.reg * self.w_output))
        d_b_output = self.step*np.sum(dE,axis = 0,keepdims=True) #bias trainig is summing the error over each point

        #adjust the weights of the hidden
        dE_hidden = np.dot(dE,self.w_output.T)
        dE_hidden[h_state <= 0 ] = 0 #eliminate the non activated neurons

        d_w_hidden = self.step*(np.dot(point_set.T,dE_hidden) + (self.reg * self.w_hidden))
        d_b_hidden = self.step*np.sum(dE_hidden,axis = 0,keepdims=True) 

        self.w_hidden -= d_w_hidden
        self.b_h -= d_b_hidden
        
        self.w_output -= d_w_output
        self.b_o -= d_b_output

         

    def classify(self,point_set):
        """
        Classifies the points

            point_set: input data set. Number of columns must match input_size parameter when the classifier is created
            
            return: vector of group membership for each point
        """
        npoints = point_set.shape[0]
        h_state = np.maximum(0,np.dot(point_set,self.w_hidden)+self.b_h)
        o_state = np.dot(h_state,self.w_output)+self.b_o
        c_vec = np.argmax(o_state,axis=1)
        c_vec.reshape((npoints,1))
        return c_vec
    


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


        

 
        

    
    


