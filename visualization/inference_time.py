import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from draw_roc import get_model_name


def read_data(csv_path: str):
    df = pd.read_csv(csv_path, header=None, sep='\t')
    df = df[list(range(len(df.columns) - 1))]
    iou, time_cost = df.iloc[:, -4], df.iloc[:, -1]
    return time_cost * 10, iou


def get_mean_and_std(data):
    data = data.values
    return np.mean(data), np.std(data)


def make_error_circles(ax, xdata, ydata, xerror, yerror, models_list, facecolor='yellow',
                       edgecolor='red', alpha=0.5):
    # Loop over data points; create box from errors at each point
    # errorcircles = [Circle((x, y), radius=random.random()) for x, y in zip(xdata, ydata)]
    for x, y, model in zip(xdata, ydata, models_list):
        ax.scatter(x, y, label=get_model_name(model))

    # # Create patch collection with specified colour/alpha
    # pc = PatchCollection(errorcircles, facecolor=facecolor, alpha=alpha,
    #                      edgecolor=edgecolor, linewidths=(1.5,))

    # # Add collection to axes
    # ax.add_collection(pc)

    # Plot errorbars
    # artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
    #                       fmt='None', ecolor='k')

    # return artists


def draw_circle_func(models_list, datasets, root='./results/metrics/'):
    for dataset in datasets:
        xdata = []
        ydata = []
        xerr = []
        yerr = []
        current_dir = os.path.join(root, f'test-{dataset}')
        for model in models_list:
            print(model)
            csv_path = os.path.join(current_dir, model + '.txt')
            x, y = read_data(csv_path)
            (x, xerror), (y, yerror) = get_mean_and_std(x), get_mean_and_std(y)
            # print(f"x:{x}, xerr: {xerror}, y: {y}, yerr: {yerror}")
            xdata.append(x)
            ydata.append(y)
            xerr.append(xerror)
            yerr.append(yerror)

        if dataset == 'TN3K':
            ydata = []
        else:
            ydata = []
        # start plot the figure
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if dataset == "TN3K":
            ax.set_title(f'{dataset} testset')
        else:
            ax.set_title(f'{dataset}')
        ax.set_xlim(9, 85)
        # ax.set_ylim(20, 40)
        ax.set_xlabel('Inference time / ms')
        ax.set_ylabel('Inference IoU / %')
        # c = list(range(1, len(models_list)+1))
        # scatter = ax.scatter(xdata, ydata, c=c)
        # handles, labels = scatter.legend_elements()
        # ax.legend(handles, models_list)
        make_error_circles(ax, xdata, ydata, xerr, yerr, models_list)
        ax.legend(loc="lower right")
        # plt.show()
        plt.savefig(os.path.join(current_dir, 'inference_time.pdf'))


if __name__ == '__main__':
    models_list = ['unet', 'sgunet', 'trfe', 'fcn', 'segnet', 'cenet', 'deeplab-v3', 'cpfnet', 'R50-ViT-B_16']
    # models_list = ['unet', 'sgunet', 'fcn', 'segnet', 'deeplab-v3', 'cenet', 'cpfnet', 'R50-ViT-B_16']
    datasets = ['TN3K', 'DDTI']
    draw_circle_func(models_list, datasets)
