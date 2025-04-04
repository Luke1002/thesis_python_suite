import argparse
import csv
import os
import time
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime as onrt

import environment_wrapper


def model_viewer(model_path, env_path, episode_id):
    session = onrt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    if os.path.basename(model_path).startswith("bc") or os.path.basename(model_path).startswith("manual_bc"):
        output_name = session.get_outputs()[0].name
    else:
        output_name = session.get_outputs()[2].name
    env = gym.make('PelletFinder-v0', env_path=env_path, worker_id=episode_id, no_graphics=True, seed=int(time.time()))
    print(f"Episode {episode_id + 1} started")
    state, _ = env.reset()
    next_actions = session.run([output_name], {input_name: state})[0]
    has_ended = False
    total_reward = 0.0
    completed_subepisodes = 0
    start = time.time()
    while not has_ended:
        state, reward, _, done, _ = env.step(next_actions)
        total_reward += reward
        if float(reward) > 1.0:
            completed_subepisodes += 1
        if not done:
            next_actions = session.run([output_name], {input_name: state})[0]
        else:
            has_ended = True
    end = time.time()
    time_elapsed = end - start
    print(f"Episode {episode_id + 1} finished in {time_elapsed} seconds")
    env.close()
    return total_reward, completed_subepisodes, time_elapsed


def complete_dataset_gatherer(model_path, env_path, num_episodes, stats):
    print(os.path.basename(model_path))
    if not (os.path.exists(env_path) and env_path.endswith(".exe")):
        print("Executable not found at path: ", env_path)
        return 2

    print("Dataset started")
    completed_subepisode = np.zeros((num_episodes, 10), dtype=float)
    rewards = np.zeros((num_episodes,), dtype=float)
    times = np.zeros((num_episodes,), dtype=float)
    for i in range(num_episodes):
        reward, completed_subepisodes, time_elapsed = model_viewer(model_path, env_path, i)
        for j in range(completed_subepisodes):
            if j < completed_subepisodes:
                completed_subepisode[i][j] += 1.0
        rewards[i] = reward
        times[i] = time_elapsed
    return rewards, times, completed_subepisode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", help="The directory where all models are stored", type=str, required=True)
    parser.add_argument("--name", "-n", help="The base name of the csv files", type=str, required=True)
    parser.add_argument("--env_path", "-e", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=500)
    args = parser.parse_args()
    stats = []
    raw_stats = []
    filename = args.name + "_"
    if os.path.basename(os.path.dirname(args.env_path)).endswith("Definitive_10"):
        stats = [["Model Name", "N models", "Mean Reward", "Mean Time", "Reaching 1 pellet", "Reaching 2 pellets",
                  "Reaching 3 pellets", "Reaching 4 pellets", "Reaching 5 pellets", "Reaching 6 pellets",
                  "Reaching 7 pellets", "Reaching 8 pellets", "Reaching 9 pellets", "Mean Success Rate"]]
        filename += f"test_stats_10_pellets_{args.num_episodes}_episodes.csv"
    elif os.path.basename(os.path.dirname(args.env_path)).endswith("Definitive_1"):
        stats = [["Model Name", "N models", "Mean Reward", "Reward Variance", "Mean Time", "Time Variance", "Mean Success Rate", "Success Rate Variance"]]
        filename += f"test_stats_1_pellet_{args.num_episodes}_episodes.csv"
    if os.path.isdir(args.model_path):
        for dir in os.listdir(args.model_path):
            dir_path = os.path.join(args.model_path, dir)
            if os.path.isdir(dir_path):
                times = np.empty(shape=(0,), dtype=float)
                rewards = np.empty(shape=(0,), dtype=float)
                success_rates = np.empty(shape=(0,10), dtype=float)
                basename = ""
                n = 0
                for file in Path(dir_path).iterdir():
                    if file.is_file() and file.suffix.lower() == ".onnx":
                        basename = os.path.basename(file)[:-7]
                        if basename.startswith("bc") or basename.startswith("manual_bc"):
                            n = int(os.path.basename(file)[-6:-5])
                            r, t, completed_subs = complete_dataset_gatherer(file, args.env_path, args.num_episodes, stats)
                            rewards = np.hstack((rewards, r))
                            times = np.hstack((times, t))
                            success_rates = np.vstack((success_rates, completed_subs))
                        else:
                            for k in range(5):
                                n = 4
                                r, t, completed_subs = complete_dataset_gatherer(file, args.env_path, args.num_episodes,
                                                                                 stats)
                                rewards = np.hstack((rewards, r))
                                times = np.hstack((times, t))
                                success_rates = np.vstack((success_rates, completed_subs))
                raw_stats.append([basename, n+1, json.dumps(np.asarray(rewards).tolist()), json.dumps(np.asarray(times).tolist()), json.dumps(np.asarray(success_rates).tolist())])
                mean_reward = rewards.mean()
                var_reward = rewards.var(ddof=1)
                mean_time = times.mean()
                var_time = times.var(ddof=1)
                success_rates = np.asarray(success_rates)
                mean_success = success_rates.mean(axis=0)
                var_success = success_rates.var(axis=0, ddof=1)
                if os.path.basename(os.path.dirname(args.env_path)).endswith("Definitive_10"):
                    stats.append([basename, n+1, mean_reward, var_reward, mean_time, var_time, mean_success[0], var_success[0]])
                elif os.path.basename(os.path.dirname(args.env_path)).endswith("Definitive_1"):
                    stats.append(
                        [basename, n+1, mean_reward, var_reward, mean_time, var_time, mean_success[0], var_success[0], mean_success[1], var_success[1], mean_success[2], var_success[2], mean_success[3], var_success[3], mean_success[4], var_success[4], mean_success[5], var_success[5], mean_success[6], var_success[6], mean_success[7], var_success[7]])
        with open(filename+".csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(raw_stats)
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(stats)