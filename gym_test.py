import gym
import snake_gym
import random
from time import sleep

env = gym.make('snake-tiled-v0')
env.reset()
while True:
    state, r, d, inf = env.step(1)
    print(state)
    sleep(5)
