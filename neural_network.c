#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void initializeNeuron(Neuron *neuron, int num_inputs) {
    neuron->bias = ((float)rand() / RAND_MAX) * 2 - 1;
    neuron->weights = (float*)malloc(num_inputs * sizeof(float));
    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}

void initializeLayer(Layer *layer, int num_neurons, int num_inputs) {
    layer->num_neurons = num_neurons;
    layer->neurons = (Neuron*)malloc(num_neurons * sizeof(Neuron));
    for (int i = 0; i < num_neurons; i++) {
        initializeNeuron(&layer->neurons[i], num_inputs);
    }
}

NeuralNetwork createNeuralNetwork(int num_input, int num_hidden1, int num_hidden2, int num_output) {
    NeuralNetwork nn;
    nn.num_input = num_input;
    initializeLayer(&nn.hidden_layer1, num_hidden1, num_input);
    initializeLayer(&nn.hidden_layer2, num_hidden2, num_hidden1);
    initializeLayer(&nn.output_layer, num_output, num_hidden2);
    return nn;
}

void forwardPass(NeuralNetwork *nn, int8_t *input, int *output) {
    for (int i = 0; i < nn->hidden_layer1.num_neurons; i++) {
        float sum = nn->hidden_layer1.neurons[i].bias;
        for (int j = 0; j < nn->num_input; j++) {
            sum += input[j] * nn->hidden_layer1.neurons[i].weights[j];
        }
        nn->hidden_layer1.neurons[i].output = sigmoid(sum);
    }

    for (int i = 0; i < nn->hidden_layer2.num_neurons; i++) {
        float sum = nn->hidden_layer2.neurons[i].bias;
        for (int j = 0; j < nn->hidden_layer1.num_neurons; j++) {
            sum += nn->hidden_layer1.neurons[j].output * nn->hidden_layer2.neurons[i].weights[j];
        }
        nn->hidden_layer2.neurons[i].output = sigmoid(sum);
    }

    for (int i = 0; i < nn->output_layer.num_neurons; i++) {
        float sum = nn->output_layer.neurons[i].bias;
        for (int j = 0; j < nn->hidden_layer2.num_neurons; j++) {
            sum += nn->hidden_layer2.neurons[j].output * nn->output_layer.neurons[i].weights[j];
        }
        nn->output_layer.neurons[i].output = sigmoid(sum);
        if (i == 0 || nn->output_layer.neurons[i].output > nn->output_layer.neurons[*output].output) {
            *output = i;
        }
    }
}

void mutateNeuralNetwork(NeuralNetwork *nn, float rate, float magnitude) {
    float mutateValue(float value, float magnitude) {
        return value + (((float)rand() / RAND_MAX) * 2 - 1) * magnitude;
    }
    
    Layer *layers[] = {&nn->hidden_layer1, &nn->hidden_layer2, &nn->output_layer};
    for (int l = 0; l < 3; l++) {
        Layer *layer = layers[l];
        for (int i = 0; i < layer->num_neurons; i++) {
            Neuron *neuron = &layer->neurons[i];
            
            if ((float)rand() / RAND_MAX < rate) {
                for (int j = 0; j < (l == 0 ? nn->num_input : layer->num_neurons); j++) {
                    neuron->weights[j] = mutateValue(neuron->weights[j], magnitude);
                }
                
                neuron->bias = mutateValue(neuron->bias, magnitude);
            }
        }
    }
}

void cleanupNeuralNetwork(NeuralNetwork *nn) {
    free(nn->hidden_layer1.neurons);
    free(nn->hidden_layer2.neurons);
    free(nn->output_layer.neurons);
}
