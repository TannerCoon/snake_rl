from re import X
import matplotlib.pyplot as plt
from IPython import display
import os

plt.ion()
# fig, ax = plt.subplots(1,1)

def plot(scores, mean_scores, total_mean_scores, title = 'Training...', fig_plot_name = None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.pause(0.1)
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(total_mean_scores)
    plt.ylim(ymin=0, ymax=max(scores)+5)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(total_mean_scores)-1, total_mean_scores[-1], str(total_mean_scores[-1]))
    if fig_plot_name:
        model_folder_path = './snake_rl/figures/'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        # fig_plot_name = os.path.join(model_folder_path, fig_plot_name)
        fig_plot_name = model_folder_path + fig_plot_name
        print(f'saving {fig_plot_name}')
        plt.savefig(fig_plot_name)
    