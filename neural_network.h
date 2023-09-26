#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

typedef struct Neuron {
    float *weights;
    float bias;
    float output;
    float delta;
} Neuron;

typedef struct Layer {
    int num_neurons;
    Neuron *neurons;
} Layer;

typedef struct NeuralNetwork {
    int num_input;
    Layer hidden_layer;
    Layer output_layer;
} NeuralNetwork;


float sigmoid(float x);
float dSigmoid(float x);
int max_element_index(float* array, int size);
void initializeNetwork(NeuralNetwork *nn, int num_input, int num_hidden_neurons, int num_output_neurons);
void forwardPropagation(NeuralNetwork *nn, float input[]);
void backwardPropagation(NeuralNetwork *nn, float target[]);
void updateWeights(NeuralNetwork *nn, float input[], float learningRate);
void trainNetwork(NeuralNetwork *nn, float inputs[][2], float targets[], int epochs, float learningRate);
void testNetwork(NeuralNetwork *nn, float inputs[][2], float targets[]);
void mutateNeuralNetwork(NeuralNetwork *nn, float rate, float magnitude);
void copyNeuralNetwork(NeuralNetwork *sourceNN, NeuralNetwork *targetNN);
void cleanupNeuralNetwork(NeuralNetwork *nn);

//save and load to and from CSV
void processNeuron(Neuron *neuron, FILE *file, int num_weights, char mode); // helper func
void saveLoadNetwork(NeuralNetwork *nn, const char *filename, char mode); // 's' to save, 'l' to load

#endif // NEURAL_NETWORK_H
