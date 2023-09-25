#include "neural_network.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <limits.h>


#define MAP_SIZE 500
#define FOOD_COUNT 2000
#define SRCH_SIZE 50
#define SNAKE_COUNT 9
#define NEURON_COUNT 5
#define EVOLVE_TIME 60000
#define RENDER_DELAY 10

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

typedef struct {
    u16 x, y;
} Point;

typedef struct {
    Point position;
    NeuralNetwork brain;
    u32 foodsEaten;
    u32 actionsSinceLastFood;
    bool touchWall;
    bool firstInit;
} Snake;


int8_t map[MAP_SIZE][MAP_SIZE] = {0};
u16 foodExisting = 0;
u16 evolutionEvents = 0;
Snake snakes[SNAKE_COUNT];
int rendering = 1;
SDL_Texture* snakeTextures[SNAKE_COUNT];
SDL_Texture* foodTexture;
bool isTextChanged = true;


// neural network architecture
int num_input = SRCH_SIZE*SRCH_SIZE;
int num_hidden1 = 6;
int num_hidden2 = 5;
int num_output = 5;

float mutationRate = 0.2;
float mutationMagnitude = 0.05;

void spawnWalls();
void spawnFoods();
void initializeSnakes();
void evolveSnakes();
void moveSnake(int s);
void updateGameLogic(u32 startTime);
bool init_SDL(SDL_Window** window, SDL_Renderer** renderer, TTF_Font** font);
void handleEvents(int* running);
void renderGame(SDL_Renderer* renderer, TTF_Font* font);
float randomFloatInRange(float range);
void saveNeuralNetworks();
void loadNeuralNetworks();


int main(void){
    srand((unsigned int)(time(NULL) + getpid()));
    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;
    TTF_Font* font = NULL;
    if(init_SDL(&window, &renderer, &font)) return 1;
    

    spawnWalls();
    spawnFoods();
    initializeSnakes();
    u32 startTime = SDL_GetTicks();
    int running = 1;
    while(running){
        handleEvents(&running);
        updateGameLogic(startTime);
        if(rendering){
            renderGame(renderer, font);
            SDL_Delay(10);
        }
    }


    // cleanup
    for(int s = 0; s < SNAKE_COUNT; s++){
        cleanupNeuralNetwork(&(snakes[s].brain));
    }
    TTF_CloseFont(font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();
    return 0;
}



void updateGameLogic(u32 startTime){
    static u32 lastEvolveTime = 0;
    u32 currentTime = SDL_GetTicks();

    if(currentTime - lastEvolveTime >= EVOLVE_TIME){
        evolveSnakes();
        lastEvolveTime = currentTime;
    }

    for(u8 s = 0; s < SNAKE_COUNT; s++){
        if(map[snakes[s].position.x][snakes[s].position.y] == 1){
            snakes[s].foodsEaten++;
            foodExisting--;
            map[snakes[s].position.x][snakes[s].position.y] = 0;
            spawnFoods();
            snakes[s].actionsSinceLastFood = 0;
        }
        moveSnake(s);
        snakes[s].actionsSinceLastFood++;
        if(snakes[s].actionsSinceLastFood > 25){
            mutateNeuralNetwork(&(snakes[s].brain), mutationRate, mutationMagnitude);
            snakes[s].actionsSinceLastFood = 0;
        }
    }
}

void initializeSnakes(){
    for(u8 s = 0; s < SNAKE_COUNT; s++){
        u16 rectWidth = round(MAP_SIZE * 0.8);
        u16 rectHeight = round(MAP_SIZE * 0.8);

        u16 minX = (MAP_SIZE - rectWidth) / 2;
        u16 minY = (MAP_SIZE - rectHeight) / 2;

        snakes[s].position.x = minX + rand() % rectWidth;
        snakes[s].position.y = minY + rand() % rectHeight;

        snakes[s].touchWall = false;
        snakes[s].foodsEaten = 0;
        snakes[s].actionsSinceLastFood = 0;

        if(!snakes[s].firstInit){
            snakes[s].brain = createNeuralNetwork(num_input, num_hidden1, num_hidden2, num_output);
            snakes[s].firstInit = true;
        }

        mutateNeuralNetwork(&(snakes[s].brain), mutationRate, mutationMagnitude);
    }
}

void evolveSnakes(){
    u8 bestSnakeIndex = 0;
    u16 maxFoodEaten = 0;
    evolutionEvents++;
    for(u8 s = 0; s < SNAKE_COUNT; s++){
        //if(!snakes[s].touchWall && snakes[s].foodsEaten > maxFoodEaten){
        if(snakes[s].foodsEaten > maxFoodEaten){
            maxFoodEaten = snakes[s].foodsEaten;
            bestSnakeIndex = s;
        }
        snakes[s].foodsEaten = 0;
    }
    if(maxFoodEaten != 0){
        for(u8 s = 0; s < SNAKE_COUNT; s++){
            if(s != bestSnakeIndex){
                copyWeights(&(snakes[s].brain), &(snakes[bestSnakeIndex].brain));
            }
        }
    }else{
        for(u8 s = 0; s < SNAKE_COUNT; s++){
            if(s != bestSnakeIndex){
                mutateNeuralNetwork(&(snakes[s].brain), mutationRate, mutationMagnitude);
            }
        }
    }
    initializeSnakes();
}

void moveSnake(int s){
    int8_t vision[SRCH_SIZE*SRCH_SIZE] = {0};
    int action = 0;

    u16 x = snakes[s].position.x, y = snakes[s].position.y;

    u16 box_x_start = MAX(0,          x - SRCH_SIZE/2),
        box_x_end   = MIN(MAP_SIZE-1, x + SRCH_SIZE/2),
        box_y_start = MAX(0,          y - SRCH_SIZE/2),
        box_y_end   = MIN(MAP_SIZE-1, y + SRCH_SIZE/2);

    u16 x_diff = x - (u16)(SRCH_SIZE / 2),
        y_diff = y - (u16)(SRCH_SIZE / 2);

    for(u16 i = box_y_start; i < box_y_end; i++){
        for(u16 j = box_x_start; j < box_x_end; j++){
            vision[(i - y_diff) * (j - x_diff)] = map[i][j];
        }
    }
    forwardPassOptimized(&(snakes[s].brain), vision, &action);

    //printf(" action snake %d : %d \n", s, action);

    switch(action){
        case 0: // do nothing
            //mutateNeuralNetwork(&(snakes[s].brain), mutationRate, mutationMagnitude);
            break;
        case 1: // up
            snakes[s].position.y = MAX(0, y - 1);
            break;
        case 2: // down
            snakes[s].position.y = MIN(MAP_SIZE - 1, y + 1);
            break;
        case 3: // left
            snakes[s].position.x = MAX(0, x - 1);
            break;
        case 4: // right
            snakes[s].position.x = MIN(MAP_SIZE - 1, x + 1);
            break;
    }
    if(snakes[s].position.y == 0 || snakes[s].position.y == (MAP_SIZE - 1) || snakes[s].position.x == 0 || snakes[s].position.x == (MAP_SIZE - 1)){
        snakes[s].touchWall = true;
        mutateNeuralNetwork(&(snakes[s].brain), mutationRate, mutationMagnitude);
    }
}

void spawnWalls(){
    for(u16 i = 0; i < MAP_SIZE; i++){
        map[0][i] = -1;
        map[MAP_SIZE-1][i] = -1;
        map[i][0] = -1;
        map[i][MAP_SIZE-1] = -1;
    }
}

void spawnFoods(){
    for(u16 i = foodExisting; i < FOOD_COUNT; i++){
        map[rand() % MAP_SIZE][rand() % MAP_SIZE] = 1;
        foodExisting++;
    }
}




void renderGame(SDL_Renderer* renderer, TTF_Font* font){
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);


    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    for(int i = 0; i < MAP_SIZE; i++){
        for(int j = 0; j < MAP_SIZE; j++){
            if(map[i][j] == 1){
                SDL_Rect foodRect = {j, i, 2, 2};
                SDL_RenderFillRect(renderer, &foodRect);
            }
        }
    }


    for(int s = 0; s < SNAKE_COUNT; s++){
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
        SDL_Rect snakeRect = { snakes[s].position.x, snakes[s].position.y, 3, 3 };
        SDL_RenderFillRect(renderer, &snakeRect);
        
        char counterText[32];
        sprintf(counterText, "S%d: %d", s + 1, snakes[s].foodsEaten);
        SDL_Color textColor = { 255, 255, 255, 120 };
        SDL_Surface* textSurface = TTF_RenderText_Solid(font, counterText, textColor);
        SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
        SDL_Rect textRect = { 10, 10 + (s * 30), textSurface->w, textSurface->h };
        SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
        SDL_FreeSurface(textSurface);
        SDL_DestroyTexture(textTexture);
    }

    char evolutionEventsText[32];
    sprintf(evolutionEventsText, "E: %d", evolutionEvents);
    SDL_Color textColor = { 255, 255, 255, 120 };
    SDL_Surface* textSurface = TTF_RenderText_Solid(font, evolutionEventsText, textColor);
    SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
    SDL_Rect textRect = { 10, 10 + (10 * 30), textSurface->w, textSurface->h };
    SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
    SDL_FreeSurface(textSurface);
    SDL_DestroyTexture(textTexture);

    char mutationRateText[32];
    sprintf(mutationRateText, "Mutation Rate: %.2f", mutationRate);
    SDL_Surface* rateSurface = TTF_RenderText_Solid(font, mutationRateText, textColor);
    SDL_Texture* rateTexture = SDL_CreateTextureFromSurface(renderer, rateSurface);
    SDL_Rect rateRect = { 10, MAP_SIZE - 60, rateSurface->w, rateSurface->h };
    SDL_RenderCopy(renderer, rateTexture, NULL, &rateRect);
    SDL_FreeSurface(rateSurface);
    SDL_DestroyTexture(rateTexture);

    char mutationMagnitudeText[32];
    sprintf(mutationMagnitudeText, "Mutation Magnitude: %.2f", mutationMagnitude);
    SDL_Surface* magnitudeSurface = TTF_RenderText_Solid(font, mutationMagnitudeText, textColor);
    SDL_Texture* magnitudeTexture = SDL_CreateTextureFromSurface(renderer, magnitudeSurface);
    SDL_Rect magnitudeRect = { 10, MAP_SIZE - 30, magnitudeSurface->w, magnitudeSurface->h };
    SDL_RenderCopy(renderer, magnitudeTexture, NULL, &magnitudeRect);
    SDL_FreeSurface(magnitudeSurface);
    SDL_DestroyTexture(magnitudeTexture);

    SDL_RenderPresent(renderer);
}

void handleEvents(int* running){
    SDL_Event e;
    while (SDL_PollEvent(&e)){
        if(e.type == SDL_QUIT){
            *running = 0;
        }
        else if(e.type == SDL_KEYDOWN){
            switch (e.key.keysym.sym){
                case SDLK_UP:
                    mutationRate += 0.01;
                    break;
                case SDLK_DOWN:
                    mutationRate = MAX(0, mutationRate - 0.01);
                    break;
                case SDLK_RIGHT:
                    mutationMagnitude += 0.01;
                    break;
                case SDLK_LEFT:
                    mutationMagnitude = MAX(0, mutationMagnitude - 0.01);
                    break;
                case SDLK_s:
                    saveNeuralNetworks();
                    break;
                case SDLK_l:
                    loadNeuralNetworks();
                    break;
                case SDLK_e:
                    evolveSnakes();
                    break;
                case SDLK_m:
                    for(u8 s = 0; s < SNAKE_COUNT; s++){
                        mutateNeuralNetwork(&(snakes[s].brain), mutationRate, mutationMagnitude);
                    }
                    break;
                case SDLK_q:
                    *running = 0;
                    break;
                case SDLK_f:
                    rendering = 0;
                    break;
                case SDLK_r:
                    rendering = 1;
                    break;
                default:
                    break;
            }
        }
    }
}



bool init_SDL(SDL_Window** window, SDL_Renderer** renderer, TTF_Font** font){
    if(SDL_Init(SDL_INIT_VIDEO) != 0){
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return true;
    }

    if(TTF_Init() != 0){
        fprintf(stderr, "Could not initialize SDL_ttf: %s\n", TTF_GetError());
        SDL_Quit();
        return true;
    }

    *window = SDL_CreateWindow("Snake Evolution Game", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, MAP_SIZE, MAP_SIZE, SDL_WINDOW_SHOWN);
    if(!(*window)){
        fprintf(stderr, "Could not create window: %s\n", SDL_GetError());
        TTF_Quit();
        SDL_Quit();
        return true;
    }

    *renderer = SDL_CreateRenderer(*window, -1, SDL_RENDERER_ACCELERATED);
    if(!(*renderer)){
        SDL_DestroyWindow(*window);
        fprintf(stderr, "Could not create renderer: %s\n", SDL_GetError());
        TTF_Quit();
        SDL_Quit();
        return true;
    }

    *font = TTF_OpenFont("Arial.ttf", 24);
    if(!(*font)){
        SDL_DestroyRenderer(*renderer);
        SDL_DestroyWindow(*window);
        fprintf(stderr, "Could not load font: %s\n", TTF_GetError());
        TTF_Quit();
        SDL_Quit();
        return true;
    }

    return false;
}


// Function to generate a random float between -range and range
float randomFloatInRange(float range){
    int randInt = rand() % 1001;
    float randFloat = (randInt / 1000.0) * (2 * range) - range;
    return randFloat;
}

void saveNeuralNetworks(){
    for(int s = 0; s < SNAKE_COUNT; s++){
        char filename[20];
        sprintf(filename, "S%d.dat", s);
        saveWeightsToFile(&(snakes[s].brain), filename);
    }
    
    printf("Neural networks saved to file.\n");
}

void loadNeuralNetworks(){    
    for(int s = 0; s < SNAKE_COUNT; s++){
        char filename[20];
        sprintf(filename, "S%d.dat", s);
        loadWeightsFromFile(&(snakes[s].brain), filename);
    }

    printf("Neural networks loaded from file.\n");
}
