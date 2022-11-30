import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from tensorboard.backend.event_processing import event_accumulator


def get_epoch_loss(event_path: str, model_name: str):
    # 加载日志数据
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    # print(ea.scalars.Keys())
    epoch_loss = ea.scalars.Items('data/epochloss')
    # print([(i.value, i.step) for i in train_loss])
    epoch_loss = [i.value for i in epoch_loss]
    return epoch_loss


def cal_mean_and_std(epoch_loss_list):
    epoch_losses = np.array(epoch_loss_list)
    mean_loss = np.mean(epoch_losses, axis=0)
    std_loss = np.std(epoch_losses, axis=0)
    return mean_loss, std_loss

def get_log_paths(model_name: str, root='./run/'):
    log_dir = os.path.join(root, model_name)
    log_paths = []
    for i in range(5):
        log_path = os.path.join(log_dir, f"fold{i}", 'log')
        file_list = os.listdir(log_path)
        sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(log_path, x)))
        sorted(file_list, key=lambda x: os.path.getsize(os.path.join(log_path, x)))
        log_paths.append(os.path.join(log_path, file_list[-1]))
    return log_paths


def draw_val_loss(ax: Axes, model_name: str):
    epoch_losses = []
    for log_file in get_log_paths(model_name):
        # print(log_file)
        epoch_losses.append(get_epoch_loss(log_file, model_name))
    y, err = cal_mean_and_std(epoch_losses)
    print(err)
    x = np.arange(len(y))
    yp = y + err
    yn = y - err
    # print(f"x: {len(x)}, y: {len(y)}, yp: {len(yp)}, yn: {len(yn)}")
    vertices = np.block([[x, x[::-1]], [yp, yn[::-1]]]).T
    codes = Path.LINETO * np.ones(len(vertices), dtype=Path.code_type)
    # vertices = np.concatenate(vertices, vertices[0])
    codes[0] = Path.MOVETO
    path = Path(vertices, codes)
    patch = PathPatch(path, facecolor='C0', edgecolor='none', alpha=0.3)
    ax.plot(x, y, label=model_name)
    ax.add_patch(patch)

def validation_loss_func(model_name: list):
    fig, ax = plt.subplots()
    ax.set_ylabel('Validation Loss')
    ax.set_xlabel('Training epoch')
    ax.set_title('TN3K segmentation', y=-0.3)
    for model_name in models_name:
        draw_val_loss(ax, model_name)
    ax.legend(loc="upper right")
    # plt.show()
    plt.savefig('./results/loss_test.pdf')


if __name__ == '__main__':
    models_name = ['trfe', 'trfeplus']
    validation_loss_func(models_name)