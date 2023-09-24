# Compiler to use
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -g -O2

# Libraries to link against
LIBS = -lSDL2 -lSDL2_ttf -lm

# Target executable
TARGET = snake_evo

# Source files
SRCS = main.c neural_network.c

# Object files
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
