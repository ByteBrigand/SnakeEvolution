#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#define GRID_SIZE 51
#define FOOD_VALUE 1.0f
#define WALL_VALUE -1.0f
#define EMPTY_VALUE 0.0f
#define NUM_SIMULATION_EVENTS 500000
#define NUM_HIDDEN_LAYER_NEURONS 4
#define DEBUGGING 1

typedef enum {
    DO_NOTHING,
    GO_UP,
    GO_DOWN,
    GO_LEFT,
    GO_RIGHT,
} Action;

float grid[GRID_SIZE][GRID_SIZE];

void initializeGrid() {
    for (int i = 0; i < GRID_SIZE; i++)
        for (int j = 0; j < GRID_SIZE; j++)
            grid[i][j] = EMPTY_VALUE;
}


void spawnFood(){
    int foodX = rand() % GRID_SIZE;
    int foodY = rand() % GRID_SIZE;
    grid[foodX][foodY] = FOOD_VALUE;
}

void spawnWalls(){
    int wallSide = rand() % 4; // Choose a random side to spawn walls
    for (int i = 0; i < GRID_SIZE; i++) {
        switch (wallSide) {
            case 0: grid[0][i] = WALL_VALUE; break; // Top
            case 1: grid[i][GRID_SIZE - 1] = WALL_VALUE; break; // Right
            case 2: grid[GRID_SIZE - 1][i] = WALL_VALUE; break; // Bottom
            case 3: grid[i][0] = WALL_VALUE; break; // Left
        }
    }
}


int calculateDistance1(int x1, int y1, int x2, int y2){
    return abs(x1 - x2) + abs(y1 - y2);
}
float calculateDistance(int x1, int y1, int x2, int y2){
    float x_diff = (float)(x1 - x2);
    float y_diff = (float)(y1 - y2);
    return sqrtf(x_diff * x_diff + y_diff * y_diff);
}

Action calculateCorrectAction() {
    int agentX = GRID_SIZE / 2;
    int agentY = GRID_SIZE / 2;
    float closestFoodDistance = GRID_SIZE * 2.0;
    Action correctAction = DO_NOTHING;

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (grid[i][j] == FOOD_VALUE) {
                float distance = calculateDistance(agentX, agentY, i, j);
                if (distance < closestFoodDistance) {
                    closestFoodDistance = distance;
                    int x_diff = i - agentX;
                    int y_diff = j - agentY;
                    if (abs(x_diff) >= abs(y_diff)) {
                        if (x_diff > 0) correctAction = GO_DOWN;
                        else if (x_diff < 0) correctAction = GO_UP;
                    } else {
                        if (y_diff > 0) correctAction = GO_RIGHT;
                        else if (y_diff < 0) correctAction = GO_LEFT;
                    }
                }
            }
        }
    }

    return correctAction;
}




void printGrid();
void printActionTaken(Action action);
int max_element_index(float* array, int size); // aka argmax


int main() {
    srand(time(NULL));
    initializeGrid();

    NeuralNetwork nn;
    initializeNetwork(&nn, GRID_SIZE * GRID_SIZE, NUM_HIDDEN_LAYER_NEURONS, 5);

    float learningRate = 0.1 ;
    int correctCount = 0;
    float totalLoss = 0;

    saveLoadNetwork(&nn, "weights.csv", 'l');

    for (int event = 0; event < NUM_SIMULATION_EVENTS; event++) {
        initializeGrid();

        for(int i = 0; i < (rand() % 20); i++)
            spawnFood();
        
        // 10% chance to spawn food next to the agent
        if (rand() % 10 == 0) {
            int agentX = GRID_SIZE / 2;
            int agentY = GRID_SIZE / 2;
            int direction = rand() % 4; // Choose a random direction: up, down, left, or right

            switch (direction) {
                case 0: // Up
                    if (agentX > 0) grid[agentX - 1][agentY] = FOOD_VALUE;
                    break;
                case 1: // Down
                    if (agentX < GRID_SIZE - 1) grid[agentX + 1][agentY] = FOOD_VALUE;
                    break;
                case 2: // Left
                    if (agentY > 0) grid[agentX][agentY - 1] = FOOD_VALUE;
                    break;
                case 3: // Right
                    if (agentY < GRID_SIZE - 1) grid[agentX][agentY + 1] = FOOD_VALUE;
                    break;
            }
        }

        float input[GRID_SIZE * GRID_SIZE];
        for (int i = 0; i < GRID_SIZE; i++)
            for (int j = 0; j < GRID_SIZE; j++)
                input[i * GRID_SIZE + j] = (float)(grid[i][j]);

        forwardPropagation(&nn, input);

        float output[5];
        for (int i = 0; i < 5; i++)
            output[i] = nn.output_layer.neurons[i].output;

        Action agentAction = (Action)(max_element_index(output, 5));
        Action correctAction = calculateCorrectAction();

        if (DEBUGGING) {
            printf("Simulation Event %d\n", event);
            printf("Agent Accuracy: %f\n", (float)correctCount / (event % 100 + 1));
            printf("Average Loss: %f\n", totalLoss / (event % 100 + 1));
            //printf("Grid:\n");
            //printGrid();
            printf("Agent Action: ");
            printActionTaken(agentAction);
            printf("\nCorrect Action: ");
            printActionTaken(correctAction);
            printf("\n\n");
        }

        if(agentAction == correctAction) correctCount++;
        float loss = 0;
        for(int i = 0; i < 5; i++) {
            float target = i == (int)correctAction ? 1.0f : 0.0f;
            loss += (target - output[i]) * (target - output[i]);
        }
        totalLoss += loss;

        // Print stats every 100 events
        if(event % 100 == 0) {
            printf("Simulation Event %d\n", event);
            printf("Agent Accuracy: %f\n", (float)correctCount / (event % 100 + 1));
            printf("Average Loss: %f\n", totalLoss / (event % 100 + 1));
            printf("\n");
            correctCount = 0;
            totalLoss = 0;
        }

        // Backpropagation and weight updates
        float target[5] = {0};
        target[correctAction] = 1.0f;
        backwardPropagation(&nn, target);
        updateWeights(&nn, input, learningRate);
    }

    saveLoadNetwork(&nn, "weights.csv", 's');

    // Cleanup
    for (int i = 0; i < nn.hidden_layer.num_neurons; i++)
        free(nn.hidden_layer.neurons[i].weights);
    free(nn.hidden_layer.neurons);

    for (int i = 0; i < nn.output_layer.num_neurons; i++)
        free(nn.output_layer.neurons[i].weights);
    free(nn.output_layer.neurons);

}











void printGrid() {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (grid[i][j] == FOOD_VALUE) {
                printf(" F "); //Food
            } else if (grid[i][j] == WALL_VALUE) {
                printf(" W "); //Wall
            } else if (i == GRID_SIZE / 2 && j == GRID_SIZE / 2) {
                printf(" A "); //Agent
            } else {
                printf(" . "); //Empty
            }
        }
        printf("\n");
    }
    printf("\n");
}

void printActionTaken(Action action) {
    switch (action) {
        case DO_NOTHING:
            printf("Nothing");
            break;
        case GO_UP:
            printf("Up");
            break;
        case GO_DOWN:
            printf("Down");
            break;
        case GO_LEFT:
            printf("Left");
            break;
        case GO_RIGHT:
            printf("Right");
            break;
        default:
            printf("Unknown");
            break;
    }
}

