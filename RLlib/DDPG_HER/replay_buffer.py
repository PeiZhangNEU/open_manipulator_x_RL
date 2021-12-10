import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        #这里的T的目的是什么：可能是为了下面的buffer存储
        self.T = env_params['max_timesteps']
        #size是总共能存episode的大小
        self.size = int(buffer_size // self.T)   # 比如1000000/200 = 5000，因为一组数据是200 * 6或者3或者4
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        print("Buffer_size:", self.size, "max_timesteps:", self.T, "env_params:", self.env_params)   # 
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),            # [5000, 201, 6]
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),            # [5000, 201, 3]
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),                 # [5000, 200, 3]
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),         # [5000, 200, 4]
                        }
        # thread lock
        self.lock = threading.Lock()  # 避免多个线程同时修改一份数据，我写单线程的程序可以忽略它

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        # print(mb_obs.shape, mb_ag.shape, mb_g.shape, mb_actions.shape)
        batch_size = mb_obs.shape[0] #batch_size 是有多少组
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)    # 得到一批idx索引，为了存放到buffer中
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]  # current_size 是0-5000之间的数，这里假设是1000
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]    # 这里也揭示了为什么obs原来是201
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]      # ag也是201，就是为了后移一格作准备
                   # ['obs',          'ag',           'g',            'actions',       'obs_next',     'ag_next']
                   # [(1000, 201, 6), (1000, 201, 3), (1000, 200, 3), (1000, 200, 4), (1000, 200, 6), (1000, 200, 3)]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)    # 从temp_buffers中用her抽取出transitions
                   # ['obs',          'ag',           'g',            'actions',       'obs_next',     'ag_next',   'r']
                   # [(64, 6),        (64, 3),        (64, 3),        (64, 4),        (64, 6),         (64, 3),     (64, 1)]
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1 #if  inc =0, inc=inc or 1  result:inc =1
        if self.current_size+inc <= self.size:  #如果buffer没有满，就直接存入，满了就替换
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
