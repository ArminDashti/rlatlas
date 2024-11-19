import minari
import numpy as np



def download(dataset_id='D4RL/pen/expert-v2'):
    dataset = minari.load_dataset(dataset_id, True)
    # return dataset
    observations, actions, rewards, terminations, truncations, next_observations = [], [], [], [], [], []
    for episode in dataset.iterate_episodes():
        episode_length = len(episode.observations)
        for i in range(100):
            observations.append(episode.observations[i])
            actions.append(episode.actions[i])
            rewards.append(episode.rewards[i]) 
            terminations.append(episode.terminations[i])
            truncations.append(episode.truncations[i])
            # next_obs = episode.observations[i + 1] if i != 199 else np.zeros_like(episode.observations[i])
            next_obs = episode.observations[i + 1]
            next_observations.append(next_obs)
    
    return dataset, {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminations': np.array(terminations),
        'truncations': np.array(truncations),
        'next_observations': np.array(next_observations),
        'dones': np.logical_or(terminations, truncations).astype(int),
    }
