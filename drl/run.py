import os
import argparse
import gym
import d3rlpy
import numpy as np
from sklearn.model_selection import train_test_split
from algorithms import BCQ, CQL, IQL
from utils import to_transitions, gta_to_data, episodes_to_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='cql')
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-replay-v0')
    parser.add_argument('--data_type', type=str, default='trim')
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--trim_pct', type=float, default=0)
    
    parser.add_argument('--training_steps', type=int, default=500000)
    parser.add_argument('--n_steps_per_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.data_type == 'trim':
        dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
        dataset = episodes_to_dataset(sorted(dataset, key = lambda episode: -episode.compute_return())[: int(len(dataset) * (1-args.trim_pct))])
    elif args.data_type == 'gta':
        import d4rl
        env = gym.make(args.dataset)
        if args.weighted:
            dataset_name = f'{args.dataset}-weighted'
        else:
            dataset_name = args.dataset
        if os.path.exists(f'./drl/gta/{dataset_name}-dataset.npz'):
            data = np.load(f'./drl/gta/{dataset_name}-dataset.npz')
        elif os.path.exists(f'./drl/gta/{dataset_name}.npz'):
            data = gta_to_data(f'./drl/gta/{dataset_name}.npz')
        else:
            raise FileNotFoundError
        dataset = to_transitions(data)        
        
    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)
    
    if args.algorithm == 'bcq':
        BCQ(args, env, dataset, test_episodes)
    elif args.algorithm == 'cql':
        CQL(args, env, dataset, test_episodes)
    else:
        IQL(args, env, dataset, test_episodes)

    
if __name__ == '__main__':
    main()
