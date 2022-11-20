from re import L
from tkinter import Y
from winsound import SND_ALIAS
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QTrainer
import numpy as np
from time import sleep


class Agent:
    def __init__(self, model, agent_params, game_params):
        self.n_games = 0
        self.epsilon = 80 # randomness
        self.gamma = 0.9 # discount rate
        self.params = agent_params
        self.game_params = game_params
        self.memory = deque(maxlen = self.params['MAX_MEMORY']) # popleft() if exceeds MAX_MEMORY
        

        self.model = model
        self.trainer = QTrainer(self.model, lr = self.params['LR'], gamma=self.gamma)


    def get_state(self, game):
        # [danger_straight, danger_right, danger_left, direction_left, direction_right, direction_up, direction_down, food_left, food_right, food_up, food_down]
        head = game.snake[0]
        snake_body = game.snake[1:]
        food = game.food
        bs = self.game_params['BLOCK_SIZE']
        w = self.game_params['WIDTH']
        h = self.game_params['HEIGHT']
        nRows = int(h/bs)
        nColumns = int(w/bs)
        if self.params['MODEL_MODE'] == 'PIXEL':
            ## Create single matrix channel
            # state = np.zeros((nRows, nColumns), dtype=int)
            # for seg in game.snake:
            #     state[int(seg.y/bs)][int(seg.x/bs)] = 0.5
            # if self.game_params['MULTI_FOOD']:
            #     for f in food:
            #         state[int(f.y/bs)][int(f.x/bs)] = 2
            # else:
            #     state[int(food.y/bs)][int(food.x/bs)] = 10
            # state = [[state]]

            ## Get Pixel values:
            # state = game.get_pixel_matricies()

            ## Create food, head and body/wall channels:
            f_grid = np.zeros((nRows+2, nColumns+2), dtype=int)
            s_grid = np.zeros((nRows+2, nColumns+2), dtype=int)
            b_grid = np.zeros((nRows+2, nColumns+2), dtype=int)
            # set walls:
            b_grid[0] = [0.5 for i in range(int(w/bs)+2)]
            b_grid[-1] = [0.5 for i in range(int(w/bs)+2)]
            for line in b_grid:
                line[0] = 0.5
                line[-1] = 0.5
            if self.game_params['MULTI_FOOD']:
                for f in food:
                    f_grid[int(f.y/bs)+1][int(f.x/bs)+1] = 10
            else:
                f_grid[int(food.y/bs)+1][int(food.x/bs)+1] = 10
            # f_grid[int(head.y/bs)+1][int(head.x/bs)+1] = 10
            s_grid[int(head.y/bs)+1][int(head.x/bs)+1] = 1
            for seg in snake_body:
                s_grid[int(seg.y/bs+1)][int(seg.x/bs+1)] = 1

            # Add additional value in front of snake head for
            # dir_l = game.direction == Direction.LEFT
            # dir_r = game.direction == Direction.RIGHT
            # dir_u = game.direction == Direction.UP
            # dir_d = game.direction == Direction.DOWN
            # try:
            #     if dir_l:
            #         h_grid[int(head.y/bs)+1][int(head.x/bs)+1-1] = 1
            #     elif dir_r:
            #         h_grid[int(head.y/bs)+1][int(head.x/bs)+1+1] = 1
            #     elif dir_u:
            #         h_grid[int(head.y/bs)+1-1][int(head.x/bs)+1] = 1
            #     else:
            #         h_grid[int(head.y/bs)+1+1][int(head.x/bs)+1] = 1
            # except:
            #     pass

            # b_grid[int(head.y/bs)+1][int(head.x/bs)+1] = 10
            
            # print('______________________________')
            # for line in grid:
            #     print(line)

            # state = np.concatenate(grid)
            state = [f_grid, s_grid, b_grid]
        else:
        # Midterm State "True_Random"
            # print([(p.x/bs, p.y/bs) for p in snake_body])
            point_l = Point(head.x - bs, head.y)
            point_r = Point(head.x + bs, head.y)
            point_u = Point(head.x, head.y - bs)
            point_d = Point(head.x, head.y + bs)

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
        if len(self.memory) > self.params['BATCH_SIZE']:
            mini_sample = random.sample(self.memory, self.params['BATCH_SIZE']) # List of tuples
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
            if self.params['RANDOM_CHOICE'] == 'TRUE_RAND':
                move = random.randint(0, 2)
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                if self.params['MODEL_MODE'] == 'PIXEL':
                    prediction = self.model(state0)[0]
                else:
                    prediction = self.model(state0)
                pred = [p + torch.abs(torch.min(prediction)).item() for p in prediction.tolist()]
                pred_pd = [i/sum(pred) for i in pred]
                move = np.random.choice([0,1,2], p = pred_pd)
        
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            if self.params['MODEL_MODE'] == 'PIXEL':
                prediction = self.model(state0)[0]
            else:
                prediction = self.model(state0)
            # print(prediction)
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        
        return final_move


def run_game(run_name, dt_str, agent_params, game_params, model, x_pos, y_pos, p, proc_scores, train = True):
    plot_scores = []
    record = 0 # best score
    agent = Agent(model, agent_params, game_params) 
    game = SnakeGameAI(game_params, x = x_pos, y = y_pos)
    while agent.n_games <= agent_params['EPISODES']:
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
    proc_scores.put((p, plot_scores))