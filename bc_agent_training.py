import os

import argparse
import time

import h5py
import numpy as np
import torch

import bc_agent_neural_network.bc_agent_model
from torch.utils.tensorboard import SummaryWriter



def get_sized_subset(dataset_path, num_episodes_to_use):
    if not os.path.exists(dataset_path):
        print("Dataset not found")
        return None
    n_episodes = num_episodes_to_use
    bc_data = np.dtype([('state', np.float32, (1, 6)), ('action', np.float32, (1, 2))])
    random_subset = []
    with h5py.File(dataset_path, "r") as f:
        tot_episodes = len(f.keys())
        if num_episodes_to_use == 0 or num_episodes_to_use > tot_episodes:
            print("Number of episodes per subset too high or 0, subset will be equal to dataset")
            n_episodes = tot_episodes
        random_indices = np.random.choice(np.arange(0, tot_episodes), size=n_episodes, replace=False)
        for i in random_indices:
            random_subset.append(np.asarray(f[str(i)], dtype=bc_data))
        f.close()
    return random_subset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", type=str, required=True)
    parser.add_argument("--length_subset", "-l", type=int, default=0)
    parser.add_argument("--num_epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    args = parser.parse_args()
    for curr_model_number in range(5):

        subset = get_sized_subset(args.dataset_path, args.length_subset)

        states = []
        actions = []
        for episode in subset:
            for step in episode:
                states.append(step["state"])
                actions.append(step["action"])
        # Extract fields as separate arrays
        states = np.squeeze(np.asarray(states))  # Shape (N, 6)
        actions = np.squeeze(np.asarray(actions))  # Shape (N, 2)
        print("Num states:", states.shape)
        print("Num actions:", actions.shape)

        states_tensor = torch.from_numpy(states).float()
        actions_tensor = torch.from_numpy(actions).float()

        model_name = f"bc_model_{len(subset)}_{args.num_epochs}_{args.batch_size}_{curr_model_number}"
        if os.path.basename(args.dataset_path).startswith("manual"):
            model_name = f"manual_{model_name}"
        start = time.time()

        writer = SummaryWriter(f"runs/{model_name}")

        policy = bc_agent_neural_network.bc_agent_model.BCAgentNetwork()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)


        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(len(states_tensor))
            state_shuffled = states_tensor[indices]
            action_shuffled = actions_tensor[indices]

            for i in range(0, len(states_tensor),  args.batch_size):
                batch_states = state_shuffled[i:i + args.batch_size]
                batch_actions = action_shuffled[i:i + args.batch_size]

                # Forward pass
                predicted_action = policy(batch_states)
                loss = criterion(predicted_action, batch_actions)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(states_tensor) / args.batch_size)

            writer.add_scalar("Loss/train", avg_loss, epoch)

            # Log model parameters
            for name, param in policy.named_parameters():
                writer.add_histogram(name, param, epoch)

            print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}")

        end = time.time()
        writer.close()
        print(f"Training completed in {end-start} s")
        example_state = torch.randn(1, 6).float()

        torch.onnx.export(
            policy,
            example_state,
            f"runs/{model_name}/{model_name}.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={"state": {0: "args.batch_size"}, "action": {0: "args.batch_size"}}
        )
