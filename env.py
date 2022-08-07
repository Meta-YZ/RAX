
from argparse import Action
from operator import ne
from jupyter_client import protocol_version_info
from pkg_resources import parse_version, working_set
import ray
import numpy as np


@ray.remote
class RayEnvWorker(object):
    def __init__(self, env, num_envs):
        self._envs = [env for _ in range(num_envs)]

    def reset(self):
        obs_ = []
        for env in self._envs:
            obs = env.reset()
            obs_.append(obs)
        return np.stack(obs_)

    def step(self, action_batch):
        rewards = []
        dones = []
        next_obs_ = []
        for env_id in range(len(self._envs)):
            next_obs, reward, done, info = self._envs[env_id].step(action_batch[env_id])
            rewards.append(reward)
            dones.append(done)
            next_obs_.append(next_obs)
        
        return {
            'rewards': np.asarray(rewards, dtype=np.float64),
            'dones': np.asarray(dones, dtype=np.bool),
            'next_obs_': np.stack(next_obs_),
        }


class RayEnvStepper(object):
    def __init__(self, env, num_envs, num_workers):
        # Put all extra envs in the worker number zero
        self._num_envs_per_worekr = [
            num_envs // num_workers if worker_index > 0 else num_envs // num_workers + num_envs % num_workers
            for worker_index in range(num_workers)
        ]
        self._workers = [
            RayEnvWorker.remote(env, num_envs=self._num_envs_per_worekr[worker_index])
            for worker_index in range(num_workers)
        ]
    
    def reset(self):
        worker_result_promises = []
        for worker_index in range(len(self._workers)):
            worker_result_promise = self._workers[worker_index].reset.remote()
            worker_result_promises.append(worker_result_promise)
        worker_results = ray.get(worker_result_promises)
        return np.concatenate(worker_results)
    
    def step(self, action_batch):
        prev_index = 0
        worker_result_promises = []
        for worker_index in range(len(self._workers)):
            worker_result_promise = self._workers[worker_index].step.remote(
                action_batch[prev_index:prev_index + self._num_envs_per_worekr[worker_index]],
            )
            worker_result_promises.append(worker_result_promise)
            prev_index += self._num_envs_per_worekr[worker_index]
        worker_results = ray.get(worker_result_promises)
        return np.concatenate(worker_results)
    
    
if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')
    batch_env = RayEnvStepper(env, num_envs=3, num_workers=2)
    obs_ = batch_env.reset()
    print(obs_)