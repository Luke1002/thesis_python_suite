import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", "-m", help="The path to the .csv file containing all statistics", type=str, required=True)
    args = parser.parse_args()

    basename = os.path.basename(args.stats_path).split('.')[0]
    plt.style.use('seaborn-v0_8-muted')
    with open(os.path.abspath(args.stats_path), newline='') as csvfile:
        stats = list(csv.reader(csvfile))
        stats = np.asarray(stats[1:])
        model_names = np.asarray(stats[:, 0])
        sorting_order = []
        for model in model_names:
            if not model.endswith('Agent'):
                ep = int(model.split('_')[-3])
                if model.startswith('manual'):
                    ep += 10000
                sorting_order.append(ep)
            else:
                sorting_order.append(np.inf)
        sorting_order = np.argsort(sorting_order)[::-1]
        model_names = model_names[sorting_order]
        y_labels =[]
        for model in model_names:
            if not model.endswith('Agent'):
                ep = int(model.split('_')[-3])
                label = f"Dataset PPO, {ep} episodi"
                if model.startswith('manual'):
                    label = f"Dataset umano, {ep} episodi"
                y_labels.append(label)
            else:
                y_labels.append("Modello di riferimento")
        mean_rewards = np.asarray(stats[:, 2], dtype=float)[sorting_order]
        var_rewards = np.asarray(stats[:, 3], dtype=float)[sorting_order]
        mean_times = np.asarray(stats[:, 4], dtype=float)[sorting_order]
        var_times = np.asarray(stats[:, 5], dtype=float)[sorting_order]
        mean_successes = np.asarray(stats[:, 6], dtype=float)[sorting_order]
        var_successes = np.asarray(stats[:, 7], dtype=float)[sorting_order]

        y_positions = np.arange(len(model_names))
        bar_height = 0.4
        error_bar_height = 0.4
        r_fig, r_ax = plt.subplots(figsize=(8, 6))


        scatter = r_ax.scatter(mean_rewards, y_positions, color='blue', label='Reward medio', zorder=3)
        for i in range(len(model_names)):
            varmin = mean_rewards[i] - var_rewards[i]
            varmax = mean_rewards[i] + var_rewards[i]
            y = y_positions[i]

            r_ax.hlines(y=y_positions[i], xmin=varmin, xmax=varmax, color='black', linewidth=1, label='Varianza' if i == 0 else "")
            r_ax.vlines(x=varmin, ymin=y - bar_height / 2, ymax=y + bar_height / 2, color='black', linewidth=1, zorder=2)
            r_ax.vlines(x=varmax, ymin=y - bar_height / 2, ymax=y + bar_height / 2, color='black', linewidth=1, zorder=2)

        r_ax.grid(axis='x', linestyle='--', alpha=0.6)
        var_line = mlines.Line2D([], [], color='black', linestyle='-', linewidth=1, label='Intervallo di varianza')
        r_ax.legend(handles=[scatter, var_line], loc='upper left')
        r_ax.set_yticks(y_positions)
        r_ax.set_yticklabels(y_labels)
        r_ax.set_xlim(0, 5)
        r_ax.set_xlabel("Reward accumulato")

        plt.savefig(basename + "_rewards.png", dpi=300, bbox_inches='tight')

        s_fig, s_ax = plt.subplots(figsize=(8, 6))

        scatter = s_ax.scatter(mean_successes, y_positions, color='blue', label='Percentuale di successo media', zorder=3)

        for i in range(len(model_names)):
            varmin = mean_successes[i] - var_successes[i]
            varmax = mean_successes[i] + var_successes[i]
            y = y_positions[i]

            s_ax.hlines(y=y_positions[i], xmin=varmin, xmax=varmax, color='black', linewidth=1, label='Varianza' if i == 0 else "")
            s_ax.vlines(x=varmin, ymin=y - bar_height / 2, ymax=y + bar_height / 2, color='black', linewidth=1, zorder=2)
            s_ax.vlines(x=varmax, ymin=y - bar_height / 2, ymax=y + bar_height / 2, color='black', linewidth=1, zorder=2)

        s_ax.grid(axis='x', linestyle='--', alpha=0.6)
        var_line = mlines.Line2D([], [], color='black', linestyle='-', linewidth=1, label='Intervallo di varianza')
        s_ax.legend(handles=[scatter, var_line], loc='upper left')
        s_ax.set_yticks(y_positions)
        s_ax.set_yticklabels(y_labels)
        s_ax.set_xlim(0, 1)
        s_ax.set_xlabel("Percentuale di successo")

        plt.savefig(basename + "_successes.png", dpi=300, bbox_inches='tight')

    with open(args.stats_path+".csv", newline='') as csvfile:
        raw_stats = np.asarray(list(csv.reader(csvfile)))
        model_names = np.asarray(raw_stats[:, 0])
        sorting_order = []
        for model in model_names:
            if not model.endswith('Agent'):
                ep = int(model.split('_')[-3])
                if model.startswith('manual'):
                    ep += 10000
                sorting_order.append(ep)
            else:
                sorting_order.append(np.inf)
        sorting_order = np.argsort(sorting_order)[::-1]
        model_names = model_names[sorting_order]
        y_labels = []
        for model in model_names:
            if not model.endswith('Agent'):
                ep = int(model.split('_')[-3])
                label = f"Dataset PPO, {ep} episodi"
                if model.startswith('manual'):
                    label = f"Dataset umano, {ep} episodi"
                y_labels.append(label)
            else:
                y_labels.append("Modello di riferimento")
        raw_rewards = np.asarray(raw_stats[:, 2])[sorting_order]
        raw_times = np.asarray(raw_stats[:, 3])[sorting_order]
        raw_succeses = np.asarray(raw_stats[:, 4])[sorting_order]

        y_positions = np.arange(len(model_names))+1
        bar_height = 0.4
        error_bar_height = 0.4
        b_fig, b_ax = plt.subplots(figsize=(8, 6))
        data_times = []
        for i in raw_times:
            clean_str = i.strip("[]").replace(",", "")
            clean_array = np.fromstring(clean_str, sep=" ")
            data_times.append(clean_array)

        b_ax.set_yticks(y_positions)
        b_ax.set_yticklabels(y_labels)
        b_ax.set_xlabel("Tempo impiegato (s)")

        boxprops = dict(color="darkblue", linewidth=2)
        flierprops = dict(marker="o", color="red", markersize=6)
        medianprops = dict(color="orange", linewidth=2)
        whiskerprops = dict(color="gray", linewidth=1.5)

        b_ax.boxplot(data_times, whis=(0, 100), vert=False,
                     boxprops=boxprops, flierprops=flierprops,
                     medianprops=medianprops, whiskerprops=whiskerprops)
        for i, data in enumerate(data_times):
            percentile_95 = float(np.percentile(data, 90))
            b_ax.vlines(x=percentile_95, ymin=y_positions[i] - bar_height/2, ymax=y_positions[i] + bar_height/2,
                        color='red', linestyle='-', linewidth=2)
        plt.yticks(y_positions, y_labels)
        b_ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        line_legend = mlines.Line2D([], [], color='red', linestyle='-', linewidth=2,
                                    label='90th Percentile')
        box_legend = mpatches.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="darkblue", linewidth=2, label='Intervallo interquartile')
        whisker_legend = mlines.Line2D([], [], color="gray", linewidth=1.5, label='Limiti')
        median_legend = mlines.Line2D([], [], color="orange", linewidth=2, label='Mediana')
        b_ax.legend(handles=[line_legend, box_legend, whisker_legend, median_legend])
        b_ax.set_xlim(0, 9.25)

        plt.savefig(basename + "_b_times.png", dpi=300, bbox_inches='tight')
