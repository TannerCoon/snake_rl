from agent import *
from model import *
# import torch
from multiprocessing import Process, Queue
from datetime import datetime
import numpy as np
import os


from helper import plot

if __name__ == '__main__':
    ### PARAMETERS:
    ## Number of simultaneous games
    PROCESS_COUNT= 6

    ## Parameters for the agent
    agent_params = {
        'EPISODES': 300,
        'LR': 0.0001,
        'BATCH_SIZE': 1000,
        'MAX_MEMORY': 100_000,
        'RANDOM_CHOICE': 'PDF', # TRUE_RAND or PDF
        'MODEL_MODE': 'OG' # PIXEL or OG
    }

    ## Parameters for the game engine
    game_params = {
        'BLOCK_SIZE': 20,
        'SPEED': 50,
        'WIDTH': 200,
        'HEIGHT': 200,
        'STEP_REWARD': 0,
        'DEATH_REWARD': -100,
        'FOOD_REWARD': 100,
        'MULTI_FOOD': False # Only use True if MODEL_MODE == PIXEL
    }

    if agent_params['MODEL_MODE'] == 'PIXEL': # Use CNN
        model = CNNet(input_channels=3, output_features=3)
        nn_params = {'input_channels': 3}
    else: # Use Basic NN
        input_layer = 11
        nn_params = {'first': input_layer, 'hidden1': 256, 'hidden2': None, 'output': 3}
        model = Linear_QNet(nn_params['first'], nn_params['hidden1'], nn_params['hidden2'], nn_params['output'])


    procs = []
    proc_scores = Queue()
    for p in range(PROCESS_COUNT):
        run_name = f"Training_{p}"
        now = datetime.now()
        dt_str = now.strftime("%Y%m%d-%H_%M_%S")
        model_folder_path = f'./model/{dt_str}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        with open(model_folder_path+'/params.txt', 'w') as pf:
            params = ['Neural Network:', str(nn_params), ' Agent:', str(agent_params)]
            pf.writelines(params)
        x_pos = (p % 6)*250 + 100
        y_pos = int(p/6)*250 + 100
        proc = Process(target=run_game, args=(run_name, dt_str, agent_params, game_params, model, x_pos, y_pos, p, proc_scores, True))
        procs.append(proc)
        proc.start()
        

    # complete the processes
    for proc in procs:
        proc.join()
    all_scores = []
    while not proc_scores.empty():
        p_id, scores = proc_scores.get()
        all_scores.append(scores)

    
    # Plot
    mean_scores = []
    tot_sum = 0
    for i in range(len(all_scores[0])):
        tot_sum += sum([all_scores[j][i] for j in range(len(all_scores))])
        mean_scores.append(tot_sum/((i+1)*len(all_scores)))

    plot(all_scores, mean_scores, fig_plot_name=f"{dt_str}_{agent_params['RANDOM_CHOICE']}_{agent_params['MODEL_MODE']}_{PROCESS_COUNT}.png")
