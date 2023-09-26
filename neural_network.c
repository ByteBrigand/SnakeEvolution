#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* structs inside neural_network.h :

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
*/


float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float dSigmoid(float x) {
    return x * (1.0 - x);
}

int max_element_index(float* array, int size) {
    int index = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] > array[index]) {
            index = i;
        }
    }
    return index;
}


void initializeNetwork(NeuralNetwork *nn, int num_input, int num_hidden_neurons, int num_output_neurons) {
    void initializeNeurons(Layer *layer, int num_weights) {
        layer->neurons = (Neuron *)malloc(layer->num_neurons * sizeof(Neuron));
        for (int i = 0; i < layer->num_neurons; i++) {
            Neuron *neuron = &layer->neurons[i];
            neuron->weights = (float *)malloc(num_weights * sizeof(float));
            neuron->bias = (float)rand() / RAND_MAX;
            for (int j = 0; j < num_weights; j++) neuron->weights[j] = (float)rand() / RAND_MAX;
        }
    }
    nn->num_input = num_input;
    nn->hidden_layer.num_neurons = num_hidden_neurons;
    nn->output_layer.num_neurons = num_output_neurons;
    
    initializeNeurons(&nn->hidden_layer, num_input);
    initializeNeurons(&nn->output_layer, num_hidden_neurons);
}




void forwardPropagation(NeuralNetwork *nn, float input[]) {
    for (int i = 0; i < nn->hidden_layer.num_neurons; i++) {
        Neuron *neuron = &nn->hidden_layer.neurons[i];
        neuron->output = 0;
        for (int j = 0; j < nn->num_input; j++) {
            neuron->output += input[j] * neuron->weights[j];
        }
        neuron->output = sigmoid(neuron->output + neuron->bias);
    }

    for (int i = 0; i < nn->output_layer.num_neurons; i++) {
        Neuron *neuron = &nn->output_layer.neurons[i];
        neuron->output = 0;
        for (int j = 0; j < nn->hidden_layer.num_neurons; j++) {
            neuron->output += nn->hidden_layer.neurons[j].output * neuron->weights[j];
        }
        neuron->output = sigmoid(neuron->output + neuron->bias);
    }
}


void backwardPropagation(NeuralNetwork *nn, float target[]) {
    for (int i = 0; i < nn->output_layer.num_neurons; i++) {
        Neuron *neuron = &nn->output_layer.neurons[i];
        float error = target[i] - neuron->output;
        neuron->delta = error * dSigmoid(neuron->output);
    }

    for (int i = 0; i < nn->hidden_layer.num_neurons; i++) {
        Neuron *neuron = &nn->hidden_layer.neurons[i];
        float error = 0;
        for (int j = 0; j < nn->output_layer.num_neurons; j++) {
            error += nn->output_layer.neurons[j].weights[i] * nn->output_layer.neurons[j].delta;
        }
        neuron->delta = error * dSigmoid(neuron->output);
    }
}

void updateWeights(NeuralNetwork *nn, float input[], float learningRate) {
    for (int i = 0; i < nn->hidden_layer.num_neurons; i++) {
        Neuron *neuron = &nn->hidden_layer.neurons[i];
        for (int j = 0; j < nn->num_input; j++) {
            neuron->weights[j] += learningRate * neuron->delta * input[j];
        }
        neuron->bias += learningRate * neuron->delta;
    }

    for (int i = 0; i < nn->output_layer.num_neurons; i++) {
        Neuron *neuron = &nn->output_layer.neurons[i];
        for (int j = 0; j < nn->hidden_layer.num_neurons; j++) {
            neuron->weights[j] += learningRate * neuron->delta * nn->hidden_layer.neurons[j].output;
        }
        neuron->bias += learningRate * neuron->delta;
    }
}

void trainNetwork(NeuralNetwork *nn, float inputs[][2], float targets[], int epochs, float learningRate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < 4; i++) {
            forwardPropagation(nn, inputs[i]);
            float target[] = {targets[i]};
            backwardPropagation(nn, target);
            updateWeights(nn, inputs[i], learningRate);
        }
    }
}

void testNetwork(NeuralNetwork *nn, float inputs[][2], float targets[]) {
    for (int i = 0; i < 4; i++) {
        forwardPropagation(nn, inputs[i]);
        printf("Input: %f, %f | Output: %f | Target: %f\n", inputs[i][0], inputs[i][1], nn->output_layer.neurons[0].output, targets[i]);
    }
}





void mutateNeuralNetwork(NeuralNetwork *nn, float rate, float magnitude) {    
    for (int i = 0; i < nn->hidden_layer.num_neurons; i++) {
        Neuron *neuron = &nn->hidden_layer.neurons[i];
        if ((float)rand() / RAND_MAX < rate) {
            neuron->bias += ((float)rand() / RAND_MAX * 2 - 1) * magnitude;
            for (int j = 0; j < nn->num_input; j++) {
                neuron->weights[j] += ((float)rand() / RAND_MAX * 2 - 1) * magnitude;
            }
        }
    }
    
    for (int i = 0; i < nn->output_layer.num_neurons; i++) {
        Neuron *neuron = &nn->output_layer.neurons[i];
        if ((float)rand() / RAND_MAX < rate) {
            neuron->bias += ((float)rand() / RAND_MAX * 2 - 1) * magnitude;
            for (int j = 0; j < nn->hidden_layer.num_neurons; j++) {
                neuron->weights[j] += ((float)rand() / RAND_MAX * 2 - 1) * magnitude;
            }
        }
    }
}



void copyNeuralNetwork(NeuralNetwork *sourceNN, NeuralNetwork *targetNN) {
    if (sourceNN->num_input != targetNN->num_input ||
        sourceNN->hidden_layer.num_neurons != targetNN->hidden_layer.num_neurons ||
        sourceNN->output_layer.num_neurons != targetNN->output_layer.num_neurons) {
        printf("Neural Networks have different architectures, cannot copy\n");
        return;
    }

    for (int i = 0; i < sourceNN->hidden_layer.num_neurons; i++) {
        Neuron *sourceNeuron = &sourceNN->hidden_layer.neurons[i];
        Neuron *targetNeuron = &targetNN->hidden_layer.neurons[i];
        targetNeuron->bias = sourceNeuron->bias;
        for (int j = 0; j < sourceNN->num_input; j++) {
            targetNeuron->weights[j] = sourceNeuron->weights[j];
        }
    }

    for (int i = 0; i < sourceNN->output_layer.num_neurons; i++) {
        Neuron *sourceNeuron = &sourceNN->output_layer.neurons[i];
        Neuron *targetNeuron = &targetNN->output_layer.neurons[i];
        targetNeuron->bias = sourceNeuron->bias;
        for (int j = 0; j < sourceNN->hidden_layer.num_neurons; j++) {
            targetNeuron->weights[j] = sourceNeuron->weights[j];
        }
    }
}

void cleanupNeuralNetwork(NeuralNetwork *nn) {
    // Cleanup hidden layer
    for (int i = 0; i < nn->hidden_layer.num_neurons; i++) {
        Neuron *neuron = &nn->hidden_layer.neurons[i];
        free(neuron->weights); // Free weights array for each neuron
    }
    free(nn->hidden_layer.neurons); // Free neurons array for the hidden layer

    // Cleanup output layer
    for (int i = 0; i < nn->output_layer.num_neurons; i++) {
        Neuron *neuron = &nn->output_layer.neurons[i];
        free(neuron->weights); // Free weights array for each neuron
    }
    free(nn->output_layer.neurons); // Free neurons array for the output layer

    // Finally, free the neural network structure itself
    free(nn);
}







void processNeuron(Neuron *neuron, FILE *file, int num_weights, char mode) {
    for (int j = 0; j < num_weights; j++) {
        mode == 's' ? fprintf(file, "%f,", neuron->weights[j]) : fscanf(file, "%f,", &neuron->weights[j]);
    }
    mode == 's' ? fprintf(file, "%f\n", neuron->bias) : fscanf(file, "%f\n", &neuron->bias);
}

void saveLoadNetwork(NeuralNetwork *nn, const char *filename, char mode) {
    FILE *file = mode == 's' ? fopen(filename, "w") : fopen(filename, "r");
    if (!file) { printf("Error opening file!\n"); return; }
    
    Layer *layers[] = {&nn->hidden_layer, &nn->output_layer};
    int num_weights[] = {nn->num_input, nn->hidden_layer.num_neurons};
    
    for (int l = 0; l < 2; l++) {
        for (int i = 0; i < layers[l]->num_neurons; i++) processNeuron(&layers[l]->neurons[i], file, num_weights[l], mode);
        fprintf(file, "\n");
    }
    
    fclose(file);
    printf("Network parameters %s to/from %s\n", mode == 's' ? "saved" : "loaded", filename);
}
