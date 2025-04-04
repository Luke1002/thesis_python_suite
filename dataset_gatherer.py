import argparse
import os
import time

import gymnasium as gym
import h5py
import keyboard
import pygame
import numpy as np
import onnxruntime as onrt

import environment_wrapper


def trained_model_data_gatherer(model_path, env_path, episode_id, no_graphics):
    session = onrt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[2].name
    env = gym.make('PelletFinder-v0', env_path=env_path, worker_id=episode_id, no_graphics=no_graphics,
                   seed=int(time.time()))

    d_episode = np.dtype([
        ('curr_state', 'f4', (1, 6)),
        ('reward', 'f4'),
        ('next_actions', 'f4', (1, 2)),
        ('done', 'b'),
        ('next_state', 'f4', (1, 6))
    ])

    start = time.time()
    print(f"Episode {episode_id + 1} started")
    episode_data = np.empty(0, dtype=d_episode)
    state, _ = env.reset()
    next_actions = session.run([output_name], {input_name: state})[0]
    reward = 0.0
    last_step = np.array((state, reward, next_actions, False, np.zeros((1, 6))), dtype=d_episode)
    has_ended = False
    while not has_ended:
        state, reward, _, done, _ = env.step(next_actions)
        last_step['next_state'] = state
        episode_data = np.append(episode_data, last_step)
        if not done:
            next_actions = session.run([output_name], {input_name: state})[0]
            last_step = np.array((state, reward, next_actions, done, np.zeros(state.shape)), dtype=d_episode)
        else:
            has_ended = True
            last_step = np.array((state, reward, np.zeros_like(next_actions), done, np.zeros_like(state)),
                                 dtype=d_episode)
            episode_data = np.append(episode_data, last_step)
    end = time.time()
    print(f"Episode {episode_id + 1} finished in {end - start} seconds")
    env.close()
    return episode_id, episode_data


def player_input_data_gatherer(env_path, episode_id):
    print("Loading model")

    env = gym.make('PelletFinder-v0', env_path=env_path, no_graphics=False, worker_id=episode_id, seed=int(time.time()))

    d_episode = np.dtype([
        ('curr_state', 'f4', (1, 6)),
        ('reward', 'f4'),
        ('next_actions', 'f4', (1, 2)),
        ('done', 'b'),
        ('next_state', 'f4', (1, 6))
    ])

    start = time.time()
    print(f"Episode {episode_id + 1} started")
    episode_data = np.empty(0, dtype=d_episode)
    state, _ = env.reset()
    next_actions = np.zeros((1, 2), dtype=float)
    reward = 0.0
    last_step = np.array((state, reward, next_actions, False, np.zeros((1, 6))), dtype=d_episode)
    has_ended = False
    while not has_ended:
        state, reward, _, done, _ = env.step(next_actions)
        last_step['next_state'] = state
        episode_data = np.append(episode_data, last_step)
        done = True if keyboard.is_pressed('enter') else done
        if not done:
            x_minus = 1.0 if keyboard.is_pressed('left') else 0.0
            x_plus = 1.0 if keyboard.is_pressed('right') else 0.0
            z_minus = 1.0 if keyboard.is_pressed('down') else 0.0
            z_plus = 1.0 if keyboard.is_pressed('up') else 0.0
            next_actions = np.array([x_plus - x_minus, z_plus - z_minus])
            last_step = np.array((state, reward, next_actions, done, np.zeros(state.shape)), dtype=d_episode)
        else:
            has_ended = True
            last_step = np.array((state, reward, np.zeros_like(next_actions), done, np.zeros_like(state)),
                                 dtype=d_episode)
            episode_data = np.append(episode_data, last_step)
    end = time.time()
    print(f"Episode {episode_id + 1} finished in {end - start} seconds")
    env.close()
    return episode_id, episode_data


def bc_optimized_trained_model_data_gatherer(model_path, env_path, episode_id, no_graphics):
    session = onrt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[2].name
    env = gym.make('PelletFinder-v0', env_path=env_path, worker_id=episode_id, no_graphics=no_graphics,
                   seed=int(time.time()))
    bc_data = np.dtype([('state', np.float32, (1, 6)), ('action', np.float32, (1, 2))])
    episode_data = np.empty(0, dtype=bc_data)
    start = time.time()
    print(f"Episode {episode_id + 1} started")
    state, _ = env.reset()
    next_actions = session.run([output_name], {input_name: state})[0]
    reward = 0.0
    has_ended = False
    while not has_ended:
        new_data = np.array([(state, next_actions)], dtype=bc_data)
        episode_data = np.append(episode_data, new_data, axis=0)
        state, reward, _, done, _ = env.step(next_actions)
        if not done:
            next_actions = session.run([output_name], {input_name: state})[0]
        else:
            has_ended = True

            new_data = np.array([(state, next_actions)], dtype=bc_data)
            episode_data = np.append(episode_data, new_data, axis=0)
    end = time.time()
    print(f"Episode {episode_id + 1} finished in {end - start} seconds")
    env.close()
    return episode_data


def bc_optimized_player_input_data_gatherer(env_path, episode_id):
    env = gym.make('PelletFinder-v0', env_path=env_path, no_graphics=False, worker_id=episode_id, seed=int(time.time()))

    bc_data = np.dtype([('state', np.float32, (1, 6)), ('action', np.float32, (1, 2))])
    episode_data = np.empty(0, dtype=bc_data)
    start = time.time()
    print(f"Episode {episode_id + 1} started")
    state, _ = env.reset()
    next_actions = np.zeros((1, 2), dtype=float)
    pygame.init()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    else:
        joystick = None
    has_ended = False
    new_data = np.array([(state, next_actions)], dtype=bc_data)
    episode_data = np.append(episode_data, new_data, axis=0)
    while not has_ended:
        new_data = np.array([(state, next_actions)], dtype=bc_data)
        if episode_data[-1] != new_data:
            episode_data = np.append(episode_data, new_data, axis=0)
        state, reward, _, done, _ = env.step(next_actions)
        done = True if keyboard.is_pressed('enter') else done
        if not done:
            x_minus = 1.0 if keyboard.is_pressed('left') else 0.0
            x_plus = 1.0 if keyboard.is_pressed('right') else 0.0
            z_minus = 1.0 if keyboard.is_pressed('down') else 0.0
            z_plus = 1.0 if keyboard.is_pressed('up') else 0.0
            next_actions = np.array([x_plus - x_minus, z_plus - z_minus])
            pygame.event.pump()
            if joystick:
                x_axis = joystick.get_axis(0)
                if np.abs(x_axis) < 0.1:
                    x_axis = 0
                z_axis = - joystick.get_axis(1)
                if np.abs(z_axis) < 0.1:
                    z_axis = 0
                joystick_actions = np.array([x_axis, z_axis])
                if np.linalg.norm(joystick_actions) > np.linalg.norm(next_actions):
                    next_actions = joystick_actions
            norm = np.linalg.norm(next_actions)
            next_actions = next_actions / norm if norm > 0 else next_actions
        else:
            has_ended = True
            new_data = np.array([(state, next_actions)], dtype=bc_data)
            episode_data = np.append(episode_data, new_data, axis=0)
    end = time.time()
    print(f"Episode {episode_id + 1} finished in {end - start} seconds")
    env.close()
    return episode_data[1:]

def complete_dataset_gatherer(dataset_name, model_path, env_path, num_episodes, no_graphics, manual_controls,
                              bc_optimized):
    start = time.time()
    print(os.path.dirname(os.path.abspath(__file__)))
    if not (os.path.exists(model_path) and model_path.endswith(".onnx")):
        print("Model not found at path: ", model_path)
        return 1
    if not (os.path.exists(env_path) and env_path.endswith(".exe")):
        print("Executable not found at path: ", env_path)
        return 2
    print("Dataset started")
    save_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(save_directory, exist_ok=True)
    if bc_optimized:
        filename = f"bc_{dataset_name}_"
    else:
        filename = f"{dataset_name}_"
    if manual_controls:
        filename = "manual_" + filename
    else:
        filename = "model_" + filename
    dataset_path = os.path.join(save_directory, filename)
    total_episodes = []
    for i in range(num_episodes):
        episode_data = np.empty(0)
        if manual_controls:
            episode_data = bc_optimized_player_input_data_gatherer(env_path,
                                                                   i) if bc_optimized else player_input_data_gatherer(
                env_path, i)
        else:
            episode_data = bc_optimized_trained_model_data_gatherer(model_path, env_path, i,
                                                                    no_graphics) if bc_optimized else trained_model_data_gatherer(
                model_path, env_path, i, no_graphics)
        total_episodes.append(episode_data)
        if i in [49, 99, 249, 499, 999, 1999]:
            dataset_path = os.path.join(save_directory, filename + f"{i + 1}.h5")
            with h5py.File(dataset_path, "w") as file:
                for j in range(len(total_episodes)):
                    file.create_dataset(f"{j}", data=total_episodes[j])

    dataset_path = os.path.join(save_directory, filename + f"{num_episodes}.h5")
    with h5py.File(dataset_path, "w") as file:
        for j in range(len(total_episodes)):
            file.create_dataset(f"{j}", data=total_episodes[j])
    end = time.time()
    if bc_optimized:
        print("Optimized Dataset saved successfully in {:.2f} seconds".format(end - start))
    else:
        print("Dataset saved successfully in {:.2f} seconds".format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_path", type=str, required=True)
    parser.add_argument("--no_graphics", type=bool, default=True)
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--manual_controls", type=bool, default=False)
    parser.add_argument("--bc_dataset", "-bc", type=bool, default=False)
    args = parser.parse_args()

    complete_dataset_gatherer(args.dataset_name, args.model_path, args.env_path, args.num_episodes,
                              args.no_graphics, args.manual_controls, args.bc_dataset)
