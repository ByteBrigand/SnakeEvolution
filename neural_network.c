#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <immintrin.h> 

float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
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

void forwardPassSimple(NeuralNetwork *nn, int8_t *input, int *output){
    for (int i = 0; i < nn->hidden_layer1.num_neurons; i++){
        float sum = nn->hidden_layer1.neurons[i].bias;
        for (int j = 0; j < nn->num_input; j++){
            sum += input[j] * nn->hidden_layer1.neurons[i].weights[j];
        }
        nn->hidden_layer1.neurons[i].output = sigmoid(sum);
    }

    for (int i = 0; i < nn->output_layer.num_neurons; i++){
        float sum = nn->output_layer.neurons[i].bias;
        for (int j = 0; j < nn->hidden_layer1.num_neurons; j++){
            sum += nn->hidden_layer1.neurons[j].output * nn->output_layer.neurons[i].weights[j];
        }
        nn->output_layer.neurons[i].output = sigmoid(sum);
        if (i == 0 || nn->output_layer.neurons[i].output > nn->output_layer.neurons[*output].output){
            *output = i;
        }
    }
}

void forwardPassOptimized(NeuralNetwork *nn, int8_t *input, int *output){
    for (int i = 0; i < nn->hidden_layer1.num_neurons; i++){
        __m128 sum_vec = _mm_set1_ps(nn->hidden_layer1.neurons[i].bias);
        for (int j = 0; j < nn->num_input; j += 4){
            __m128 input_vec = _mm_loadu_ps((float *)&input[j]);
            __m128 weight_vec = _mm_loadu_ps(&nn->hidden_layer1.neurons[i].weights[j]);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(input_vec, weight_vec));
        }
        nn->hidden_layer1.neurons[i].output = sigmoid(_mm_cvtss_f32(sum_vec));
    }

    for (int i = 0; i < nn->output_layer.num_neurons; i++){
        __m128 sum_vec = _mm_set1_ps(nn->output_layer.neurons[i].bias);
        for (int j = 0; j < nn->hidden_layer1.num_neurons; j++){
            __m128 hidden_output_vec = _mm_set1_ps(nn->hidden_layer1.neurons[j].output);
            __m128 weight_vec = _mm_loadu_ps(&nn->output_layer.neurons[i].weights[j]);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(hidden_output_vec, weight_vec));
        }
        nn->output_layer.neurons[i].output = sigmoid(_mm_cvtss_f32(sum_vec));
        if (i == 0 || nn->output_layer.neurons[i].output > nn->output_layer.neurons[*output].output){
            *output = i;
        }
    }
}



void mutateNeuralNetwork(NeuralNetwork *nn, float rate, float magnitude){
    float mutateValue(float value, float magnitude){
        return value + (((float)rand() / RAND_MAX) * 2 - 1) * magnitude;
    }
    
    void normalizeWeights(float *weights, int size){
        float min = FLT_MAX;
        float max = FLT_MIN;
        for(int i = 0; i < size; i++){
            if(weights[i] < min){
                min = weights[i];
            }
            if(weights[i] > max){
                max = weights[i];
            }
        }
        for(int i = 0; i < size; i++){
            weights[i] = (weights[i] - min) / (max - min);
        }
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
                
                normalizeWeights(neuron->weights, (l == 0 ? nn->num_input : layer->num_neurons));
                
                neuron->bias = mutateValue(neuron->bias, magnitude);
            }
        }
    }
}


int cleanupNeuralNetwork(NeuralNetwork *nn) {
    if(nn == NULL){
        fprintf(stderr, "Neural network pointer is NULL. Cannot clean up.\n");
        return -1;
    }
    int err = 0;

    #define CLEANUP_LAYER(layer) \
        if(nn->layer.neurons != NULL){ \
            for(int i = 0; i < nn->layer.num_neurons; i++){ \
                if(nn->layer.neurons[i].weights != NULL){ \
                    free(nn->layer.neurons[i].weights); \
                    nn->layer.neurons[i].weights = NULL; \
                }else{ \
                    fprintf(stderr, "Weights of neuron %d in " #layer " are NULL.\n", i); \
                    err = -1; \
                } \
            } \
            free(nn->layer.neurons); \
            nn->layer.neurons = NULL; \
        }else{ \
            fprintf(stderr, "Neurons in " #layer " are NULL.\n"); \
            err = -1; \
        }

    CLEANUP_LAYER(hidden_layer1);
    CLEANUP_LAYER(hidden_layer2);
    CLEANUP_LAYER(output_layer);

    #undef CLEANUP_LAYER

    return err;
}



#define COPY_LAYER_WEIGHTS(LAYER, SIZE) \
    for(int i = 0; i < nn1->LAYER.num_neurons; i++){ \
        nn2->LAYER.neurons[i].bias = nn1->LAYER.neurons[i].bias; \
        for(int j = 0; j < SIZE; j++){ \
            nn2->LAYER.neurons[i].weights[j] = nn1->LAYER.neurons[i].weights[j]; \
        } \
    }

void copyWeights(NeuralNetwork *nn1, NeuralNetwork *nn2){
    if(nn1 == NULL || nn2 == NULL){
        fprintf(stderr, "One of the Neural Network pointers is NULL.\n");
        return;
    }

    COPY_LAYER_WEIGHTS(hidden_layer1, nn1->num_input);
    COPY_LAYER_WEIGHTS(hidden_layer2, nn1->hidden_layer1.num_neurons);
    COPY_LAYER_WEIGHTS(output_layer, nn1->hidden_layer2.num_neurons);
}

int saveWeightsToFile(NeuralNetwork *nn, const char *filename){
    FILE *file = fopen(filename, "wb");
    if(file == NULL){
        fprintf(stderr, "Unable to open file for writing: %s\n", filename);
        return -1;
    }

    Layer *layers[] = {&nn->hidden_layer1, &nn->hidden_layer2, &nn->output_layer};
    for(int l = 0; l < 3; l++){
        Layer *layer = layers[l];
        for(int i = 0; i < layer->num_neurons; i++){
            Neuron *neuron = &layer->neurons[i];
            fwrite(neuron->weights, sizeof(float), l == 0 ? nn->num_input : layer->num_neurons, file);
            fwrite(&neuron->bias, sizeof(float), 1, file);
        }
    }

    fclose(file);
    return 0;
}

int loadWeightsFromFile(NeuralNetwork *nn, const char *filename){
    FILE *file = fopen(filename, "rb");
    if(file == NULL){
        fprintf(stderr, "Unable to open file for reading: %s\n", filename);
        return -1;
    }

    Layer *layers[] = {&nn->hidden_layer1, &nn->hidden_layer2, &nn->output_layer};
    for(int l = 0; l < 3; l++){
        Layer *layer = layers[l];
        for(int i = 0; i < layer->num_neurons; i++){
            Neuron *neuron = &layer->neurons[i];
            fread(neuron->weights, sizeof(float), l == 0 ? nn->num_input : layer->num_neurons, file);
            fread(&neuron->bias, sizeof(float), 1, file);
        }
    }

    fclose(file);
    return 0;
}



