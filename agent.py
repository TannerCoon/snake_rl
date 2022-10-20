from re import L
from tkinter import Y
from winsound import SND_ALIAS
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot
from datetime import datetime
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EPISODES = 200

class Agent:
    def __init__(self, nn_params, train = True, model_path = None):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if exceeds MAX_MEMORY
        if train:
            self.model = Linear_QNet(nn_params['first'], nn_params['hidden1'], nn_params['hidden2'], nn_params['output'])
        else:
            self.model = Linear_QNet(nn_params['first'], nn_params['hidden1'], nn_params['hidden2'], nn_params['output'],model_path)
        self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)


    def get_state(self, game):
        # [danger_straight, danger_right, danger_left, direction_left, direction_right, direction_up, direction_down, food_left, food_right, food_up, food_down]
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
        ]
        
        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memeory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample) # Extracts the lists from the tuples in memory
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, train = True):
        # random moves: tradeoff between exploration / exploitation
        if train:
            self.epsilon = 80 - self.n_games # can adjust the 80 to change randomness rate
        else:
            self.epsilon = 0
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


def run(run_name, dt_str, episodes, nn_params, train = True, model_path = None):
    plot_scores = []
    plot_mean_scores = []
    plot_total_mean_scores = []
    total_score = 0
    record = 0 # best score
    if train:
        agent = Agent(nn_params) 
    else:
        agent = Agent(nn_params, train, model_path)
    game = SnakeGameAI()
    if train:
        plot_title = 'Training...'
    else:
        plot_title = 'Testing...'
    while agent.n_games <= episodes:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, train)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if train:
        # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            if train:
                agent.train_long_memeory()

            if score > record:
                record = score
                game.record = score
                if train:
                    agent.model.save(dt_str, run_name, record)
            
            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

            plot_scores.append(score)
            total_score += score
            if len(plot_scores) > 10:
                mean_score = sum(plot_scores[-10:]) / 10
            else:
                mean_score = sum(plot_scores) / len(plot_scores)
            plot_mean_scores.append(mean_score)
            total_mean_score = total_score / agent.n_games
            plot_total_mean_scores.append(total_mean_score)
            plot(plot_scores, plot_mean_scores, plot_total_mean_scores, plot_title)
    plot(plot_scores, plot_mean_scores, plot_total_mean_scores, fig_plot_name=f"{run_name}_{dt_str}.png")
    if train:
        file_name = f'{run_name}__{record}.pth'
        model_folder_path = f'./snake_rl/model/{dt_str}'
        model_path = os.path.join(model_folder_path, file_name)
    else:

        model_path = None
    return model_path




if __name__ == '__main__':
    model_params = [{'first': 11, 'hidden1': 256, 'hidden2': None, 'output': 3}, 
                    {'first': 11, 'hidden1': 64, 'hidden2': None, 'output': 3}, 
                    {'first': 11, 'hidden1': 64, 'hidden2': 64, 'output': 3}, 
                    {'first': 11, 'hidden1': 256, 'hidden2':256, 'output': 3}]
    for nn_params in model_params:
        train_name = "Training"
        test_name = "Testing"
        now = datetime.now()
        dt_str = now.strftime("%Y%m%d-%H_%M_%S")

        model_folder_path = f'./snake_rl/model/{dt_str}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        with open(model_folder_path+'/params.txt', 'w') as pf:
            params = ['Neural Network:', str(nn_params), 'Leaning Rate', str(LR)]
            pf.writelines(params)
        # run(test_name, dt_str, 50, nn_params, train=False, model_path='./snake_rl/model/20221011-17_31_33/Training__63.pth')
        model_path = run(train_name, dt_str, EPISODES, nn_params, train=True)
        run(test_name, dt_str, 50, nn_params, train=False, model_path=model_path)
    
