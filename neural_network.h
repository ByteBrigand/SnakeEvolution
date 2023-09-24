#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include <stdbool.h>

typedef struct Neuron {
    float *weights;
    float bias;
    float output;
} Neuron;

typedef struct Layer {
    int num_neurons;
    Neuron *neurons;
} Layer;

typedef struct NeuralNetwork {
    int num_input;
    Layer hidden_layer1;
    Layer hidden_layer2;
    Layer output_layer;
} NeuralNetwork;

NeuralNetwork createNeuralNetwork(int num_input, int num_hidden1, int num_hidden2, int num_output);
void forwardPass(NeuralNetwork *nn, int8_t *input, int *output);
void mutateNeuralNetwork(NeuralNetwork *nn, float rate, float magnitude);
void cleanupNeuralNetwork(NeuralNetwork *nn);

#endif // NEURAL_NETWORK_H
