import re
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import os
import random



pygame.init()
# screen = pygame.display.set_mode((100,100)) 
font = pygame.font.Font('./arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 00)
DARK_GREEN = (0,255,100)
BLACK = (0,0,0)

class SnakeGameAI:
    
    # def __init__(self, w=640, h=480):
    def __init__(self, game_params, x = random.randint(500,2000), y = random.randint(500,1000)):
        self.params = game_params
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)
        self.grid_size = (self.params['WIDTH']/self.params['BLOCK_SIZE']) * (self.params['HEIGHT']/self.params['BLOCK_SIZE'])
        self.game_n = 0
        self.record = 0
        # init display
        self.display = pygame.display.set_mode((self.params['WIDTH'], self.params['HEIGHT']))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.params['WIDTH']/2, self.params['HEIGHT']/2)
        self.snake = [self.head, 
                      Point(self.head.x-self.params['BLOCK_SIZE'], self.head.y),
                      Point(self.head.x-(2*self.params['BLOCK_SIZE']), self.head.y)]
        
        self.score = 0
        # self.food = None
        self.food = []
        self._place_food()
        self.game_n += 1
        self.frame_iteration = 0
        self.steps = self.grid_size * 2
        self.steps_since_food = 0

    def _set_food_piece(self):
        x = random.randint(0, (self.params['WIDTH']-self.params['BLOCK_SIZE'] )//self.params['BLOCK_SIZE'] )*self.params['BLOCK_SIZE'] 
        y = random.randint(0, (self.params['HEIGHT']-self.params['BLOCK_SIZE'] )//self.params['BLOCK_SIZE'] )*self.params['BLOCK_SIZE']
        f = Point(x, y)
        if f in self.snake or any([f in food for food in self.food]):
            f = self._set_food_piece()
        return f
        
    def _place_food(self):
        if self.params['MULTI_FOOD']:
            self.food = []
            x = random.randint(0, (self.params['WIDTH']-self.params['BLOCK_SIZE'] )//self.params['BLOCK_SIZE'] )*self.params['BLOCK_SIZE'] 
            y = random.randint(0, (self.params['HEIGHT']-self.params['BLOCK_SIZE'] )//self.params['BLOCK_SIZE'] )*self.params['BLOCK_SIZE']
            for i in range(max(1, int((200 - self.game_n)/10))):
                self.food.append(self._set_food_piece())
        else:
            x = random.randint(0, (self.params['WIDTH']-self.params['BLOCK_SIZE'] )//self.params['BLOCK_SIZE'] )*self.params['BLOCK_SIZE'] 
            y = random.randint(0, (self.params['HEIGHT']-self.params['BLOCK_SIZE'] )//self.params['BLOCK_SIZE'] )*self.params['BLOCK_SIZE']
            self.food = Point(x, y)
            if self.food in self.snake:
                self._place_food()

        
    def play_step(self, action):
        self.frame_iteration += 1
        reward = self.params['STEP_REWARD']
        self.steps -= 1
        self.steps_since_food += 1 
        # 1. collect user input
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()
        
        # 2. move
        prev_head = self.snake[0]
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        ## Movement reward
        # prev_diff = abs(prev_head.x - self.food.x) + abs(prev_head.y - self.food.y)
        # cur_diff = abs(self.snake[0].x - self.food.x) + abs(self.snake[0].y - self.food.y)
        # if prev_diff > cur_diff:
        #     # print('closer')
        #     reward = 1
        # else:
        #     reward = -1
        #    # print('further')
        # print(prev_head, self.snake[0], self.food)
        # 3. check if game over
        
        game_over = False
        if self.is_collision() or self.steps <= 0 or self.steps_since_food >= 150: # or (self.game_n < 50 and self.steps_since_food >= 15):
            game_over = True
            reward = self.params['DEATH_REWARD']
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.params['MULTI_FOOD']:
            if any([self.head == f for f in self.food]):
                self.score += 1
                reward = self.params['FOOD_REWARD']
                self._place_food()
                self.steps += self.grid_size
                self.steps_since_food = 0
            else:
                self.snake.pop()
        else:
            if self.head == self.food:
                self.score += 1
                reward = self.params['FOOD_REWARD']
                self._place_food()
                self.steps += self.grid_size
                self.steps_since_food = 0
            else:
                self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.params['SPEED'])
        # 6. return reward, game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.params['WIDTH'] - self.params['BLOCK_SIZE'] or pt.x < 0 or pt.y > self.params['HEIGHT'] - self.params['BLOCK_SIZE'] or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.head.x, self.head.y, self.params['BLOCK_SIZE'], self.params['BLOCK_SIZE']))
        for pt in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.params['BLOCK_SIZE'], self.params['BLOCK_SIZE']))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        if self.params['MULTI_FOOD']:
            for f in self.food:
                pygame.draw.rect(self.display, RED, pygame.Rect(f.x, f.y, self.params['BLOCK_SIZE'], self.params['BLOCK_SIZE']))
        else:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.params['BLOCK_SIZE'], self.params['BLOCK_SIZE']))
        
        text = font.render(f"Game: {str(self.game_n)}, Score: {str(self.score)}, Record: {str(self.record)}", True, WHITE)
        # self.display.blit(text, [0, 0])
        pygame.display.flip()
        # print(pygame.PixelArray(pygame.display.get_surface()).shape)
        
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u -^
        else: # [0,0,1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d -^

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.params['BLOCK_SIZE']
        elif self.direction == Direction.LEFT:
            x -= self.params['BLOCK_SIZE']
        elif self.direction == Direction.DOWN:
            y += self.params['BLOCK_SIZE']
        elif self.direction == Direction.UP:
            y -= self.params['BLOCK_SIZE']
            
        self.head = Point(x, y)

    def get_pixel_matricies(self):
        # print(pygame.PixelArray(pygame.display.get_surface()))
        red_channel = pygame.surfarray.array_red(pygame.display.get_surface())
        green_channel = pygame.surfarray.array_green(pygame.display.get_surface())
        blue_channel = pygame.surfarray.array_blue(pygame.display.get_surface())

        return [red_channel, green_channel, blue_channel]


