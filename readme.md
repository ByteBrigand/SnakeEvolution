# Snake Evolution Game

Snake Evolution Game is a simple evolutionary simulation.
Snakes' neural networks are not trained using machine learning techniques but rather evolve over time through genetic algorithms and natural selection based on their performance in finding and consuming food.
Since it can take a very long time to see some intelligence, a simulator is included that allows for the use of backpropagation.

## Features

- Multiple snakes with their own neural networks.
- Evolution of neural networks over time.
- Adjustable mutation rate and mutation magnitude.
- Save and load neural networks from a file.
- Visual representation of the game using graphics.

## Dependencies

- SDL2 (Simple DirectMedia Layer 2)
- SDL2_ttf (SDL TrueType Font library)

### Installation on debian (run as root or with sudo)

   ```bash
   apt-get update
   apt-get install libsdl2-dev libsdl2-ttf-dev
   ```


## Controls

- Use the arrow keys to adjust the mutation rate and mutation magnitude.
- Press "s" to save the neural networks to a file.
- Press "l" to load neural networks from a file.
- Press "e" to manually evolve the snakes.
- Press "q" to quit the game.
- Press "f" to pause rendering.
- Press "r" to resume rendering.

## License

This project is licensed under the GPL2 License.