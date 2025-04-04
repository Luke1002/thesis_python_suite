import os.path

import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", "-m", help="The path to the .csv file containing all statistics", type=str, required=True)
    args = parser.parse_args()

    with open(os.path.abspath(args.stats_path), newline='') as csvfile:
        stats = list(csv.reader(csvfile))
        stats = np.asarray(stats[1:])
        model_names = np.asarray(stats[:, 0])
        sorting_order = []
        for model in model_names:
            if not model.endswith('Agent'):
                ep = int(model.split('_')[-2])
                sorting_order.append(ep)
            else:
                sorting_order.append(np.inf)
        sorting_order = np.argsort(sorting_order)[::-1]
        model_names = model_names[sorting_order]
        mean_rewards = np.asarray(stats[:, 2], dtype=float)[sorting_order]
        var_rewards = np.asarray(stats[:, 3], dtype=float)[sorting_order]
        mean_times = np.asarray(stats[:, 4], dtype=float)[sorting_order]
        var_times = np.asarray(stats[:, 5], dtype=float)[sorting_order]
        mean_successes = np.asarray(stats[:, 6], dtype=float)[sorting_order]
        var_successes = np.asarray(stats[:, 7], dtype=float)[sorting_order]

        y_positions = np.arange(len(model_names))
        bar_height = 0.4
        error_bar_height = 0.4

        r_fig, r_ax = plt.subplots()
        r_ax.barh(y_positions, mean_rewards, height=bar_height, color='lightgray', edgecolor='gray')
        for i in range(len(model_names)):
            varmin = mean_rewards[i] - var_rewards[i]
            varmax = mean_rewards[i] + var_rewards[i]

            r_ax.hlines(y=y_positions[i], xmin=varmin, xmax=varmax, color='black', linewidth=1)

            r_ax.vlines(x=varmin, ymin=y_positions[i] - error_bar_height, ymax=y_positions[i] + error_bar_height,
                        color='black', linewidth=1)
            r_ax.vlines(x=varmax, ymin=y_positions[i] - error_bar_height, ymax=y_positions[i] + error_bar_height,
                        color='black', linewidth=1)

        r_ax.set_yticks(y_positions)
        r_ax.set_yticklabels(model_names)
        r_ax.set_xlabel("Reward accumulato")
        r_ax.set_title("Confronto tra modelli")

        plt.savefig("rewards.png", dpi=600, bbox_inches='tight')

        t_fig, t_ax = plt.subplots()
        t_ax.barh(y_positions, mean_times, height=bar_height, color='lightgray', edgecolor='gray')
        for i in range(len(model_names)):
            varmin = mean_times[i] - var_times[i]
            varmax = mean_times[i] + var_times[i]

            t_ax.hlines(y=y_positions[i], xmin=varmin, xmax=varmax, color='black', linewidth=1)
            t_ax.vlines(x=varmin, ymin=y_positions[i] - error_bar_height, ymax=y_positions[i] + error_bar_height,
                        color='black', linewidth=1)
            t_ax.vlines(x=varmax, ymin=y_positions[i] - error_bar_height, ymax=y_positions[i] + error_bar_height,
                        color='black', linewidth=1)

        t_ax.set_yticks(y_positions)
        t_ax.set_yticklabels(model_names)
        t_ax.set_xlabel("Tempo impiegato (s)")

        plt.savefig("times.png", dpi=600, bbox_inches='tight')

        s_fig, s_ax = plt.subplots()
        s_ax.barh(y_positions, mean_successes, height=bar_height, color='lightgray', edgecolor='gray')
        for i in range(len(model_names)):
            varmin = mean_successes[i] - var_successes[i]
            varmax = mean_successes[i] + var_successes[i]

            s_ax.hlines(y=y_positions[i], xmin=varmin, xmax=varmax, color='black', linewidth=1)
            s_ax.vlines(x=varmin, ymin=y_positions[i] - error_bar_height, ymax=y_positions[i] + error_bar_height,
                        color='black', linewidth=1)
            s_ax.vlines(x=varmax, ymin=y_positions[i] - error_bar_height, ymax=y_positions[i] + error_bar_height,
                        color='black', linewidth=1)

        s_ax.set_yticks(y_positions)
        s_ax.set_yticklabels(model_names)
        s_ax.set_xlabel("Probabilità di succcesso")
        s_ax.set_title("Confronto tra modelli")

        plt.savefig("successes.png", dpi=600, bbox_inches='tight')

    with open(args.stats_path+".csv", newline='') as csvfile:
        raw_stats = np.asarray(list(csv.reader(csvfile)))
        model_names = np.asarray(raw_stats[:, 0])
        sorting_order = []
        for model in model_names:
            if not model.endswith('Agent'):
                ep = int(model.split('_')[-2])
                sorting_order.append(ep)
            else:
                sorting_order.append(np.inf)
        sorting_order = np.argsort(sorting_order)[::-1]
        model_names = model_names[sorting_order]
        raw_rewards = np.asarray(raw_stats[:, 2])[sorting_order]
        raw_times = np.asarray(raw_stats[:, 3])[sorting_order]
        raw_succeses = np.asarray(raw_stats[:, 4])[sorting_order]

        y_positions = np.arange(len(model_names))+1
        bar_height = 0.4
        error_bar_height = 0.4
        b_fig, b_ax = plt.subplots()
        data_rewards = []
        b_ax.set_yticks(y_positions)
        b_ax.set_yticklabels(model_names)
        b_ax.set_xlabel("Probabilità di succcesso")
        for i in raw_rewards:
            clean_str = i.strip("[]").replace(",", "")
            clean_array = np.fromstring(clean_str, sep=" ")
            data_rewards.append(clean_array)

        plt.boxplot(data_rewards, whis=(25, 75), vert=False)

        plt.yticks(y_positions, model_names)

        plt.savefig("test_b_times.png", dpi=600, bbox_inches='tight')
