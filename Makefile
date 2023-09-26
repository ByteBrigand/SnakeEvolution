# Compiler to use
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -O2

# Libraries to link against
LIBS = -lSDL2 -lSDL2_ttf -lm -msse4.2

# Source files for snake_evo
SRCS_SNAKE_EVO = main.c neural_network.c

# Source files for sim
SRCS_SIM = sim.c neural_network.c

# Object files for snake_evo
OBJS_SNAKE_EVO = $(SRCS_SNAKE_EVO:.c=.o)

# Object files for sim
OBJS_SIM = $(SRCS_SIM:.c=.o)

# Target executables
TARGET_SNAKE_EVO = snake_evo
TARGET_SIM = sim

all: $(TARGET_SNAKE_EVO) $(TARGET_SIM)

$(TARGET_SNAKE_EVO): $(OBJS_SNAKE_EVO)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(TARGET_SIM): $(OBJS_SIM)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS_SNAKE_EVO) $(OBJS_SIM) $(TARGET_SNAKE_EVO) $(TARGET_SIM)
