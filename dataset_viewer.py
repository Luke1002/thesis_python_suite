import argparse
import os
import time

import gymnasium as gym
import onnxruntime as onrt

import environment_wrapper

def model_viewer(model_path, env_path, episode_id):

    session = onrt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    env = gym.make('PelletFinder-v0', env_path=env_path, worker_id=episode_id, no_graphics=False, seed = int(time.time()))
    start = time.time()
    print(f"Episode {episode_id+1} started")
    state, _ = env.reset()
    next_actions = session.run([output_name], {input_name: state})[0]
    has_ended = False
    while not has_ended:
        state, reward, _, done, _ = env.step(next_actions)
        if not done:
            next_actions = session.run([output_name], {input_name: state})[0]
        else:
            has_ended = True
    end = time.time()
    print(f"Episode {episode_id+1} finished in {end - start} seconds")
    env.close()



def complete_dataset_gatherer(model_path, env_path, num_episodes):
    start = time.time()
    print(os.path.dirname(os.path.abspath(__file__)))
    if not (os.path.exists(os.path.abspath(model_path)) and model_path.endswith(".onnx")):
        print("Model not found at path: ", os.path.abspath(model_path))
        return 1
    if not (os.path.exists(env_path) and env_path.endswith(".exe")):
        print("Executable not found at path: ", env_path)
        return 2
    print("Dataset started")
    for i in range(num_episodes):
        model_viewer(model_path, env_path, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_path", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=500)
    args = parser.parse_args()

    complete_dataset_gatherer(args.model_path, args.env_path, args.num_episodes)