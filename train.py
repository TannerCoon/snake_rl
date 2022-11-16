from agent import *
from model import *
# import torch
from multiprocessing import Process, Queue
from datetime import datetime
import numpy as np
import os


from helper import plot

if __name__ == '__main__':
    # model_params = [{'first': 11, 'hidden1': 256, 'hidden2': None, 'output': 3}, 
    #                 {'first': 11, 'hidden1': 64, 'hidden2': None, 'output': 3}, 
    #                 {'first': 11, 'hidden1': 64, 'hidden2': 64, 'output': 3}, 
    #                 {'first': 11, 'hidden1': 256, 'hidden2':256, 'output': 3}]
    
    # if train:
    #     self.model = Linear_QNet(nn_params['first'], nn_params['hidden1'], nn_params['hidden2'], nn_params['output'])
    # else:
    #     self.model = Linear_QNet(nn_params['first'], nn_params['hidden1'], nn_params['hidden2'], nn_params['output'],model_path)
    
    # proc = Process(target=print_func)  # instantiating without any argument
    # procs.append(proc)
    # proc.start()
    PROCESS_COUNT= 6
    agent_params = {
        'EPISODES': 300,
        'LR': 0.0001,
        'BATCH_SIZE': 1000,
        'MAX_MEMORY': 100_000,
        'RANDOM_CHOICE': 'PDF', # TRUE_RAND or PDF
        'MODEL_MODE': 'PIXEL' # PIXEL or OG
    }
    game_params = {
        'BLOCK_SIZE': 20,
        'SPEED': 80,
        'WIDTH': 200,
        'HEIGHT': 200,
        'STEP_REWARD': 0,
        'DEATH_REWARD': -100,
        'FOOD_REWARD': 100,
        'MULTI_FOOD': False
    }

    if agent_params['MODEL_MODE'] == 'PIXEL':
        model = CNNet(input_channels=3, output_features=3)
        nn_params = {'input_channels': 3}
    #     input_layer = int((int(game_params['WIDTH']/game_params['BLOCK_SIZE'])+2) * (int(game_params['HEIGHT']/game_params['BLOCK_SIZE'])+2))
    #     nn_params = {'first': input_layer, 'hidden1': 256, 'hidden2': 64, 'output': 3}
    else:
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
        #model_path = run()
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

    
    # mean_scores = list(np.mean(all_scores, axis=0))
    mean_scores = []
    tot_sum = 0
    for i in range(len(all_scores[0])):
        # mean_scores.extend()
        tot_sum += sum([all_scores[j][i] for j in range(len(all_scores))])
        mean_scores.append(tot_sum/((i+1)*len(all_scores)))
    # print(mean_scores)
    # print(all_scores)
    # print(mean_scores)
    plot(all_scores, mean_scores, fig_plot_name=f"{dt_str}_{agent_params['RANDOM_CHOICE']}_{agent_params['MODEL_MODE']}_{PROCESS_COUNT}.png")

    #plot(plot_scores, plot_mean_scores, plot_total_mean_scores, fig_plot_name=f"{run_name}_{dt_str}.png")


    # for nn_params in [model_params[0]]:
    #     train_name = "Training"
    #     test_name = "Testing"
    #     now = datetime.now()
    #     dt_str = now.strftime("%Y%m%d-%H_%M_%S")

    #     model_folder_path = f'./model/{dt_str}'
    #     if not os.path.exists(model_folder_path):
    #         os.makedirs(model_folder_path)
    #     with open(model_folder_path+'/params.txt', 'w') as pf:
    #         params = ['Neural Network:', str(nn_params), 'Leaning Rate', str(LR)]
    #         pf.writelines(params)
    #     # run(test_name, dt_str, 50, nn_params, train=False, model_path='./model/20221011-17_31_33/Training__63.pth')
    #     model_path = run(train_name, dt_str, EPISODES, nn_params, train=True)
    #     # run(test_name, dt_str, 50, nn_params, train=False, model_path=model_path)