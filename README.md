# Simple Neural Network

## Description
Python implemenation of basic building blocks for neural networks

This module  can be used to learn about how neural networks work. Please feel free to modify it for your own use.
Its not intended to be used for any hardcore deep learning, and not robust enough(yet) to handle erroneous usage.
It is however fully functional and can be used to train any neural networks, albeit slower than a C implementation


Uses the following objects that can be used to programmatically build neural networks:

 - `InputSource`: single input source to the neural network
 - `LayerBias`: sinlge input source to the neural network that allways has a value of 1.0. Used for biasing a neuron output
 - `OutputSink`: hooked to the output of a single neuron, used to compute the value of that neuron and train it
 - `Neuron`: main Neuron object

The module additional provides several activation function. The activation functions are in the form 

    act_Fun(val,dv=False)

where val is the value, and the dv is a flag that is set to False to evaluate the value of activation function, 
or set to true to evaluate the derivative of the activation function used for back propagation

The way to create a neural network using this module is:

1. Set up the `InputSource` objects and a single Bias ojbec
2. Create your hidden layer with an array of `Neuron` objects.
3. For each neuron of the hidden layer, use the `add_child` function to add as the inputs and bias object 
   as children with inital weights
4. Repeat steps 2 and 3 for each layer, including the last layer
5. Create `OutputSink` objects for every output neuron that you wish to read output on. 

Computing the value of the network is done by calling `compute_output` on the `OutputSink` object
Training the network through 1 epoch with backpropagation is done by calling `train_children(target)` on every `OutputSink` object. 

Call the former 2 things in a loop, keeping track of the error to get the network to converge on a target

## Requirements

    python3
    numpy

## Example

run `python3 nn_function_estimator.py` to see an example of a neural network training itself to represent a function. 
Requires `matplotlib`
