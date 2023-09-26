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


#define GRID_SIZE 500
#define WALL_SHIFT 5
#define FOOD_VALUE 1.0f
#define WALL_VALUE -1.0f
#define EMPTY_VALUE 0.0f
#define FOOD_COUNT 2000
#define SRCH_SIZE 51
#define SNAKE_COUNT 9
#define EVOLVE_TIME 10000
#define RENDER_DELAY 10
#define NUM_HIDDEN_LAYER_NEURONS 4
#define DEBUGGING 1

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    int x, y;
} Point;

typedef struct {
    Point position;
    NeuralNetwork brain;
    int foodsEaten;
    int actionsSinceLastFood;
    bool touchWall;
    bool firstInit;
} Snake;

typedef enum {
    DO_NOTHING,
    GO_UP,
    GO_DOWN,
    GO_LEFT,
    GO_RIGHT,
} Action;


float grid[GRID_SIZE][GRID_SIZE];
Snake snakes[SNAKE_COUNT];
Point* foodArray = NULL;
Point changedFoodCoord;
int foodExisting = 0;
int evolutionEvents = 0;
int rendering = 1;
SDL_Texture* snakeTextures[SNAKE_COUNT];
SDL_Texture* foodTexture;
bool areWallsDrawn = false;
bool isFoodDrawnInitial = false;
bool isTextChanged = true;
bool isFoodChanged = true;

char prevCounterText[SNAKE_COUNT][32];
char prevEvolutionEventsText[32];
char prevMutationRateText[32];
char prevMutationMagnitudeText[32];


// neural network architecture
int num_input = SRCH_SIZE*SRCH_SIZE;
int num_hidden1 = NUM_HIDDEN_LAYER_NEURONS;
int num_output = 5;

float mutationRate = 0.1;
float mutationMagnitude = 0.01;


// function prototypes
void initializeGrid();
bool checkSnakeOnFood(int x, int y);
void updateGameLogic();
void initializeSnakes();
void evolveSnakes();
bool snakeTakeAction(int s, Action act);
void extractROI(float vision[], int x, int y);
void processSnake(int s);
bool checkMoveValid(int x, int y);
void pushFood(int x, int y);
Point popFood();
void eatFood(int x, int y);
void spawnFood(int x, int y);
void spawnFoods();
void spawnWalls();
void renderText(SDL_Renderer* renderer, TTF_Font* font, const char* text, int x, int y, SDL_Color textColor);
bool stringChanged(const char* str1, const char* str2);
void renderGame(SDL_Renderer* renderer, TTF_Font* font);
void handleEvents(int* running);
bool init_SDL(SDL_Window** window, SDL_Renderer** renderer, TTF_Font** font);
float randomFloatInRange(float range);
void manageNeuralNetworks(char action);


int main(void){
    srand((unsigned int)(time(NULL) + getpid()));
    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;
    TTF_Font* font = NULL;
    if(init_SDL(&window, &renderer, &font)) return 1;
    
    initializeGrid();
    spawnWalls();
    spawnFoods();
    initializeSnakes();

    int running = 1;
    while(running){
        handleEvents(&running);
        updateGameLogic();
        if(rendering){
            renderGame(renderer, font);
            SDL_Delay(10);
        }
    }


    // cleanup
    for(int s = 0; s < SNAKE_COUNT; s++){
        cleanupNeuralNetwork(&snakes[s].brain);
    }
    TTF_CloseFont(font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();
    return 0;
}

void initializeGrid(){
    for (int i = 0; i < GRID_SIZE; i++)
        for (int j = 0; j < GRID_SIZE; j++)
            grid[i][j] = EMPTY_VALUE;
}


bool checkSnakeOnFood(int x, int y){
    return grid[y][x] == FOOD_VALUE;
}


void updateGameLogic(){
    static int lastEvolveTime = 0;
    int currentTime = SDL_GetTicks();

    if(currentTime - lastEvolveTime >= EVOLVE_TIME){
        evolveSnakes();
        lastEvolveTime = currentTime;
    }

    for(int s = 0; s < SNAKE_COUNT; s++){
        int x = snakes[s].position.x;
        int y = snakes[s].position.y;
        if(checkSnakeOnFood(x,y)){
            eatFood(x,y);
            spawnFoods();
            snakes[s].foodsEaten++;
            snakes[s].actionsSinceLastFood = 0;
        }
        processSnake(s);
        if(snakes[s].actionsSinceLastFood++ > 25){
            mutateNeuralNetwork(&snakes[s].brain, mutationRate, mutationMagnitude);
            snakes[s].actionsSinceLastFood = 0;
        }
    }
}

void initializeSnakes(){
    for(int s = 0; s < SNAKE_COUNT; s++){
        int rectWidth = (int)(GRID_SIZE * 0.75);
        int rectHeight = (int)(GRID_SIZE * 0.75);

        int minX = (int)((GRID_SIZE - rectWidth) / 2);
        int minY = (int)((GRID_SIZE - rectHeight) / 2);

        snakes[s].position.x = minX + rand() % rectWidth;
        snakes[s].position.y = minY + rand() % rectHeight;

        snakes[s].touchWall = false;
        snakes[s].foodsEaten = 0;
        snakes[s].actionsSinceLastFood = 0;

        if(!snakes[s].firstInit){
            initializeNetwork(&snakes[s].brain, num_input, num_hidden1, num_output);
            snakes[s].firstInit = true;
            saveLoadNetwork(&snakes[s].brain, "weights.csv", 'l');
        }

        mutateNeuralNetwork(&snakes[s].brain, mutationRate, mutationMagnitude);
    }
}

void evolveSnakes(){
    int bestSnakeIndex = 0;
    int maxFoodEaten = 0;
    evolutionEvents++;
    for(int s = 0; s < SNAKE_COUNT; s++){
        //if(!snakes[s].touchWall && snakes[s].foodsEaten > maxFoodEaten){
        if(snakes[s].foodsEaten > maxFoodEaten){
            maxFoodEaten = snakes[s].foodsEaten;
            bestSnakeIndex = s;
        }
        snakes[s].foodsEaten = 0;
    }

    for(int s = 0; s < SNAKE_COUNT; s++){
        if(s != bestSnakeIndex){
            if(maxFoodEaten != 0){
                copyNeuralNetwork(&snakes[bestSnakeIndex].brain, &snakes[s].brain);
            }else{
                mutateNeuralNetwork(&snakes[s].brain, mutationRate, mutationMagnitude);
            }
        }
    }

    initializeSnakes();
}

bool snakeTakeAction(int s, Action act){ // true if good action
    int x = snakes[s].position.x;
    int y = snakes[s].position.y;
    int new_x = x;
    int new_y = y;
    switch(act){
        case DO_NOTHING:
            //mutateNeuralNetwork(&snakes[s].brain, mutationRate, mutationMagnitude);
            return 0;
            break;
        case GO_UP:    new_y--; break;
        case GO_DOWN:  new_y++; break;
        case GO_LEFT:  new_x--; break;
        case GO_RIGHT: new_x++; break;
    }
    if(checkMoveValid(new_x, new_y)){
        snakes[s].position.x = new_x;
        snakes[s].position.y = new_y;
        return 1;
    }else{
        snakes[s].touchWall = true;
        return 0;
    }
}

void extractROI(float vision[], int x, int y){
    memset(vision, 0, SRCH_SIZE * SRCH_SIZE * sizeof(float));
    int box_x_start = (int)MAX(0, x - SRCH_SIZE / 2.0);
    int box_x_end   = (int)MIN(GRID_SIZE - 1, x + SRCH_SIZE / 2.0);
    int box_y_start = (int)MAX(0, y - SRCH_SIZE / 2.0);
    int box_y_end   = (int)MIN(GRID_SIZE - 1, y + SRCH_SIZE / 2.0);

    //int x_diff = x - (int)(SRCH_SIZE / 2.0);
    //int y_diff = y - (int)(SRCH_SIZE / 2.0);

    int vision_index = 0;

    for (int i = box_y_start; i < box_y_end; i++){
        for (int j = box_x_start; j < box_x_end; j++){
            vision[vision_index] = grid[i][j];
            vision_index++;
        }
    }
}


void processSnake(int s){
    float vision[SRCH_SIZE*SRCH_SIZE] = {0.0f};

    int x = snakes[s].position.x;
    int y = snakes[s].position.y;

    extractROI(vision, x, y);

    forwardPropagation(&snakes[s].brain, vision);

    float output[5];
    for (int i = 0; i < 5; i++)
        output[i] = snakes[s].brain.output_layer.neurons[i].output;

    Action agentAction = (Action)(max_element_index(output, 5));

    if(!snakeTakeAction(s, agentAction)){
        mutateNeuralNetwork(&snakes[s].brain, mutationRate, mutationMagnitude);
    }
}

bool checkMoveValid(int x, int y){ // true if valid
    const int WALL_END = GRID_SIZE - WALL_SHIFT;
    return !(x < 0 || y < 0 || x >= GRID_SIZE || y >= GRID_SIZE || x == WALL_SHIFT || y == WALL_SHIFT || x >= WALL_END || y >= WALL_END);
}

void pushFood(int x, int y){
    foodExisting++;
    foodArray = (Point*)realloc(foodArray, foodExisting * sizeof(Point));
    if (foodArray == NULL){
        perror("Memory allocation error");
        exit(1);
    }
    foodArray[foodExisting - 1].x = x;
    foodArray[foodExisting - 1].y = y;
}

Point popFood(){
    if (foodExisting == 0){
        Point emptyPoint = { -1, -1 }; // no food available
        return emptyPoint;
    }

    Point poppedFood = foodArray[foodExisting - 1];
    foodExisting--;

    if (foodExisting == 0){
        free(foodArray);
        foodArray = NULL;
    } else {
        foodArray = (Point*)realloc(foodArray, foodExisting * sizeof(Point));
        if (foodArray == NULL){
            perror("Memory allocation error");
            exit(1);
        }
    }
    return poppedFood;
}

void eatFood(int x, int y){
    popFood(x,y);
    grid[y][x] = EMPTY_VALUE;
    isFoodChanged = true;
}

void spawnFood(int x, int y){
    pushFood(x,y);
    grid[y][x] = FOOD_VALUE;
    isFoodChanged = true;
}

void spawnFoods(){
    #define RANDOM_COORD() (WALL_SHIFT + 1 + rand() % (GRID_SIZE - 2 - WALL_SHIFT))
    for(int i = foodExisting; i < FOOD_COUNT; i++){
        spawnFood(RANDOM_COORD(),RANDOM_COORD());
    }
}

void spawnWalls(){
    int size = GRID_SIZE-WALL_SHIFT;
    for(int i = WALL_SHIFT; i < size; i++){
        grid[WALL_SHIFT][i] = grid[size-1][i] = grid[i][WALL_SHIFT] = grid[i][size-1] = WALL_VALUE;
    }
}




void renderText(SDL_Renderer* renderer, TTF_Font* font, const char* text, int x, int y, SDL_Color textColor){
    SDL_Surface* textSurface = TTF_RenderText_Solid(font, text, textColor);
    SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
    SDL_Rect textRect = { x, y, textSurface->w, textSurface->h };
    SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
    SDL_FreeSurface(textSurface);
    SDL_DestroyTexture(textTexture);
}

bool stringChanged(const char* str1, const char* str2){
    return strcmp(str1, str2) != 0;
}

void renderGame(SDL_Renderer* renderer, TTF_Font* font){
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_Color textColor = {255, 255, 255, 120};

    // walls
    if(!areWallsDrawn || DEBUGGING){
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 150);
        for (int i = 0; i < GRID_SIZE; i++){
            for (int j = 0; j < GRID_SIZE; j++){
                if (grid[i][j] == WALL_VALUE){
                    SDL_Rect wallRect = { j, i, 1, 1 };
                    SDL_RenderFillRect(renderer, &wallRect);
                }
            }
        }
    }

    // food
    if(!isFoodDrawnInitial || DEBUGGING){
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        for (int i = 0; i < foodExisting; i++){
            SDL_Rect foodRect = { foodArray[i].y, foodArray[i].x, 2, 2 };
            SDL_RenderFillRect(renderer, &foodRect);
        }
        isFoodDrawnInitial = true;
    }
    if(isFoodChanged){
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        SDL_Rect foodRect = { changedFoodCoord.y, changedFoodCoord.x, 2, 2 };
        SDL_RenderFillRect(renderer, &foodRect);
        isFoodChanged = false;
    }
    
    // snakes
    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
    for (int s = 0; s < SNAKE_COUNT; s++){
        SDL_Rect snakeRect = { snakes[s].position.x, snakes[s].position.y, 3, 3 };
        SDL_RenderFillRect(renderer, &snakeRect);

        char counterText[32];
        sprintf(counterText, "S%d: %d", s + 1, snakes[s].foodsEaten);
        if (stringChanged(counterText, prevCounterText[s]) || DEBUGGING){
            renderText(renderer, font, counterText, 10, 10 + (s * 30), textColor);
            strcpy(prevCounterText[s], counterText);
        }
    }

    char evolutionEventsText[32];
    sprintf(evolutionEventsText, "Evo: %d", evolutionEvents);
    if (stringChanged(evolutionEventsText, prevEvolutionEventsText) || DEBUGGING){
        renderText(renderer, font, evolutionEventsText, 10, GRID_SIZE - 90, textColor);
        strcpy(prevEvolutionEventsText, evolutionEventsText);
    }

    char mutationRateText[32];
    sprintf(mutationRateText, "Mutation Rate: %.2f", mutationRate);
    if (stringChanged(mutationRateText, prevMutationRateText) || DEBUGGING){
        renderText(renderer, font, mutationRateText, 10, GRID_SIZE - 60, textColor);
        strcpy(prevMutationRateText, mutationRateText);
    }

    char mutationMagnitudeText[32];
    sprintf(mutationMagnitudeText, "Mutation Magnitude %.2f", mutationMagnitude);
    if (stringChanged(mutationMagnitudeText, prevMutationMagnitudeText) || DEBUGGING){
        renderText(renderer, font, mutationMagnitudeText, 10, GRID_SIZE - 30, textColor);
        strcpy(prevMutationMagnitudeText, mutationMagnitudeText);
    }

    SDL_RenderPresent(renderer);
}


void handleEvents(int* running){
    SDL_Event e;
    while (SDL_PollEvent(&e)){
        if (e.type == SDL_QUIT){
            *running = 0;
        } else if (e.type == SDL_KEYDOWN){
            switch (e.key.keysym.sym){
                case SDLK_UP: mutationRate += 0.01; isTextChanged = true; break;
                case SDLK_DOWN: mutationRate = MAX(0, mutationRate - 0.01); isTextChanged = true; break;
                case SDLK_RIGHT: mutationMagnitude += 0.01; isTextChanged = true; break;
                case SDLK_LEFT: mutationMagnitude = MAX(0, mutationMagnitude - 0.01); isTextChanged = true; break;
                case SDLK_s: manageNeuralNetworks('s'); break;
                case SDLK_l: manageNeuralNetworks('l'); break;
                case SDLK_e: evolveSnakes(); break;
                case SDLK_m:
                    for (int s = 0; s < SNAKE_COUNT; s++){
                        mutateNeuralNetwork(&snakes[s].brain, mutationRate, mutationMagnitude);
                    }
                    break;
                case SDLK_q: *running = 0; break;
                case SDLK_f: rendering = 0; break;
                case SDLK_r: rendering = 1; break;
                default: break;
            }
        }
    }
}




bool init_SDL(SDL_Window** window, SDL_Renderer** renderer, TTF_Font** font){
    if (SDL_Init(SDL_INIT_VIDEO) != 0){
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return true;
    }

    if (TTF_Init() != 0){
        fprintf(stderr, "Could not initialize SDL_ttf: %s\n", TTF_GetError());
        goto cleanup_sdl;
    }

    *window = SDL_CreateWindow("Snake Evolution Game", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, GRID_SIZE, GRID_SIZE, SDL_WINDOW_SHOWN);
    if (!*window){
        fprintf(stderr, "Could not create window: %s\n", SDL_GetError());
        goto cleanup_ttf;
    }

    *renderer = SDL_CreateRenderer(*window, -1, SDL_RENDERER_ACCELERATED);
    if (!*renderer){
        fprintf(stderr, "Could not create renderer: %s\n", SDL_GetError());
        goto cleanup_window;
    }

    *font = TTF_OpenFont("Arial.ttf", 24);
    if (!*font){
        fprintf(stderr, "Could not load font: %s\n", TTF_GetError());
        goto cleanup_renderer;
    }

    return false;

cleanup_renderer:
    SDL_DestroyRenderer(*renderer);
cleanup_window:
    SDL_DestroyWindow(*window);
cleanup_ttf:
    TTF_Quit();
cleanup_sdl:
    SDL_Quit();
    return true;
}



// random float between -range and range
float randomFloatInRange(float range){
    return ((float)rand() / RAND_MAX) * (2 * range) - range;
}

void manageNeuralNetworks(char action){
    for (int s = 0; s < SNAKE_COUNT; s++){
        char filename[20];
        sprintf(filename, "weights_S%d.dat", s);

        if (action == 's'){
            saveLoadNetwork(&snakes[s].brain, filename, 's');
        } else if (action == 'l'){
            saveLoadNetwork(&snakes[s].brain, filename, 'l');
        }
    }
}

