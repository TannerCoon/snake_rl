# snake_rl 
#### GitHub Link: https://github.com/TannerCoon/snake_rl
Reinforcement Learning For Snake Game 

credit to: https://www.youtube.com/watch?v=PJl4iabBEz0
           https://github.com/python-engineer/snake-ai-pytorch
           https://github.com/maurock/snake-ga

## Recommended setup 
Create and activate python3 environment specific for this project using the command 'python -m venv snake_rl'.

Install the packages in the requirements.txt to it with pip.

Run the commands from the parent directory above the cloned repository.

## How to run the AI driven game
run:
python3 ./train.py

## Code components

### snake_game-human.py
Launches a basic snake game that takes the directional keys on the keyboard as inputs to move the snake around the screen.

### game.py
Modification of the snake_game-human.py code that allows the movements of the snake to be controlled by an AI agent.

### agent.py
Logic for interpreting the game state. It then feeds that state information to a nueral network model outlined in the model.py code. Then it recieves an action to be performed as an output from that model and conforms that into an movement in the game.

### model.py
Structure for the reinforcement learning, neural network model that inputs game state and outputs actions.

### helper.py
Helper function used to plot the score of each game, the mean of the last 10 games, and the mean of all games in a line chart.

## Additional components

### arial.tff
Text format used to show score and additional information in the snake game window.

### figures/
Directory that contains saved images of the Training and Testing score plots.

### model/
Directoy that contains the saved training models that obtained the record score during the training episodes. You will also find a params.txt within each timestamped subdirectory that identifies the hyperparameters used in the model for training.
