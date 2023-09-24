#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float sigmoid(float x){
    return 1.0 / (1.0 + exp(-x));
}

void initializeNeuron(Neuron *neuron, int num_inputs){
    neuron->bias = ((float)rand() / RAND_MAX) * 2 - 1;
    neuron->weights = (float*)malloc(num_inputs * sizeof(float));
    for(int i = 0; i < num_inputs; i++){
        neuron->weights[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}

void initializeLayer(Layer *layer, int num_neurons, int num_inputs){
    layer->num_neurons = num_neurons;
    layer->neurons = (Neuron*)malloc(num_neurons * sizeof(Neuron));
    for(int i = 0; i < num_neurons; i++){
        initializeNeuron(&layer->neurons[i], num_inputs);
    }
}

NeuralNetwork createNeuralNetwork(int num_input, int num_hidden1, int num_hidden2, int num_output){
    NeuralNetwork nn;
    nn.num_input = num_input;
    initializeLayer(&nn.hidden_layer1, num_hidden1, num_input);
    initializeLayer(&nn.hidden_layer2, num_hidden2, num_hidden1);
    initializeLayer(&nn.output_layer, num_output, num_hidden2);
    return nn;
}

void forwardPass(NeuralNetwork *nn, int8_t *input, int *output){
    for(int i = 0; i < nn->hidden_layer1.num_neurons; i++){
        float sum = nn->hidden_layer1.neurons[i].bias;
        for(int j = 0; j < nn->num_input; j++){
            sum += input[j] * nn->hidden_layer1.neurons[i].weights[j];
        }
        nn->hidden_layer1.neurons[i].output = sigmoid(sum);
    }

    for(int i = 0; i < nn->hidden_layer2.num_neurons; i++){
        float sum = nn->hidden_layer2.neurons[i].bias;
        for(int j = 0; j < nn->hidden_layer1.num_neurons; j++){
            sum += nn->hidden_layer1.neurons[j].output * nn->hidden_layer2.neurons[i].weights[j];
        }
        nn->hidden_layer2.neurons[i].output = sigmoid(sum);
    }

    for(int i = 0; i < nn->output_layer.num_neurons; i++){
        float sum = nn->output_layer.neurons[i].bias;
        for(int j = 0; j < nn->hidden_layer2.num_neurons; j++){
            sum += nn->hidden_layer2.neurons[j].output * nn->output_layer.neurons[i].weights[j];
        }
        nn->output_layer.neurons[i].output = sigmoid(sum);
        if(i == 0 || nn->output_layer.neurons[i].output > nn->output_layer.neurons[*output].output){
            *output = i;
        }
    }
}

void mutateNeuralNetwork(NeuralNetwork *nn, float rate, float magnitude){
    float mutateValue(float value, float magnitude){
        return value + (((float)rand() / RAND_MAX) * 2 - 1) * magnitude;
    }
    
    Layer *layers[] = {&nn->hidden_layer1, &nn->hidden_layer2, &nn->output_layer};
    for(int l = 0; l < 3; l++){
        Layer *layer = layers[l];
        for(int i = 0; i < layer->num_neurons; i++){
            Neuron *neuron = &layer->neurons[i];
            
            if((float)rand() / RAND_MAX < rate){
                for(int j = 0; j < (l == 0 ? nn->num_input : layer->num_neurons); j++){
                    neuron->weights[j] = mutateValue(neuron->weights[j], magnitude);
                }
                
                neuron->bias = mutateValue(neuron->bias, magnitude);
            }
        }
    }
}

int cleanupNeuralNetwork(NeuralNetwork *nn){
    if (nn == NULL) {
        fprintf(stderr, "Neural network pointer is NULL. Cannot clean up.\n");
        return -1;
    }

    int err = 0;

    // Clean up hidden layer 1
    if (nn->hidden_layer1.neurons != NULL) {
        for(int i = 0; i < nn->hidden_layer1.num_neurons; i++){
            if (nn->hidden_layer1.neurons[i].weights != NULL) {
                free(nn->hidden_layer1.neurons[i].weights);
                nn->hidden_layer1.neurons[i].weights = NULL;
            } else {
                fprintf(stderr, "Weights of neuron %d in hidden layer 1 are NULL.\n", i);
                err = -1;
            }
        }
        free(nn->hidden_layer1.neurons);
        nn->hidden_layer1.neurons = NULL;
    } else {
        fprintf(stderr, "Neurons in hidden layer 1 are NULL.\n");
        err = -1;
    }

    // Clean up hidden layer 2
    if (nn->hidden_layer2.neurons != NULL) {
        for(int i = 0; i < nn->hidden_layer2.num_neurons; i++){
            if (nn->hidden_layer2.neurons[i].weights != NULL) {
                free(nn->hidden_layer2.neurons[i].weights);
                nn->hidden_layer2.neurons[i].weights = NULL;
            } else {
                fprintf(stderr, "Weights of neuron %d in hidden layer 2 are NULL.\n", i);
                err = -1;
            }
        }
        free(nn->hidden_layer2.neurons);
        nn->hidden_layer2.neurons = NULL;
    } else {
        fprintf(stderr, "Neurons in hidden layer 2 are NULL.\n");
        err = -1;
    }

    // Clean up output layer
    if (nn->output_layer.neurons != NULL) {
        for(int i = 0; i < nn->output_layer.num_neurons; i++){
            if (nn->output_layer.neurons[i].weights != NULL) {
                free(nn->output_layer.neurons[i].weights);
                nn->output_layer.neurons[i].weights = NULL;
            } else {
                fprintf(stderr, "Weights of neuron %d in output layer are NULL.\n", i);
                err = -1;
            }
        }
        free(nn->output_layer.neurons);
        nn->output_layer.neurons = NULL;
    } else {
        fprintf(stderr, "Neurons in output layer are NULL.\n");
        err = -1;
    }

    return err;
}


void copyWeights(NeuralNetwork *nn1, NeuralNetwork *nn2) {
    if (nn1 == NULL || nn2 == NULL) {
        fprintf(stderr, "One of the Neural Network pointers is NULL.\n");
        return;
    }
    
    // Copy weights and biases from hidden layer 1
    for(int i = 0; i < nn1->hidden_layer1.num_neurons; i++) {
        nn2->hidden_layer1.neurons[i].bias = nn1->hidden_layer1.neurons[i].bias;
        for(int j = 0; j < nn1->num_input; j++) {
            nn2->hidden_layer1.neurons[i].weights[j] = nn1->hidden_layer1.neurons[i].weights[j];
        }
    }
    
    // Copy weights and biases from hidden layer 2
    for(int i = 0; i < nn1->hidden_layer2.num_neurons; i++) {
        nn2->hidden_layer2.neurons[i].bias = nn1->hidden_layer2.neurons[i].bias;
        for(int j = 0; j < nn1->hidden_layer1.num_neurons; j++) {
            nn2->hidden_layer2.neurons[i].weights[j] = nn1->hidden_layer2.neurons[i].weights[j];
        }
    }
    
    // Copy weights and biases from output layer
    for(int i = 0; i < nn1->output_layer.num_neurons; i++) {
        nn2->output_layer.neurons[i].bias = nn1->output_layer.neurons[i].bias;
        for(int j = 0; j < nn1->hidden_layer2.num_neurons; j++) {
            nn2->output_layer.neurons[i].weights[j] = nn1->output_layer.neurons[i].weights[j];
        }
    }
}


