import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]         #时间步数， 每回合时间步数200
        rollout_batch_size = episode_batch['actions'].shape[0]    #batch 有多少rollout  也就是episode tempbuffer有多少条数据
        batch_size = batch_size_in_transitions        # 抽样batchsize 64
        ''' select which rollouts and which timesteps to be used
        #  np.random.randint(low, high=None, size=None, dtype='l') low:生成元素的最小值
        # high:生成元素的值一定小于high值
        # size:输出的大小，可以是整数也可以是元组
        # dtype:生成元素的数据类型
        # 注意：high不为None，生成元素的值在[low,high)区间中；如果high=None，生成的区间为[0,low)区间
        '''
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)  # 从1000抽64个数
        t_samples = np.random.randint(T, size=batch_size)                    # 从200抽取64个数
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()} # 从tempbuffer中抽取64组数据.
                                                                     # ['obs',    'ag',    'g',  'actions','obs_next','ag_next']
                                                                     # [(64, 6), (64, 3), (64, 3), (64, 4), (64, 6), (64, 3)]
        # her idx
        #np.where: 按条件筛选元素
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p) #产生0-1的随机数,从64个序号中抽出小于0.9的index，数目不定，假设为50个
        #np.random.uniform(size=batch_size) 150个0-1
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)    # 产生64个0-1的随机数，对应乘以 200减去64个序号
        future_offset = future_offset.astype(int)                               # 对上面的运算结果取整数
        future_t = (t_samples + 1 + future_offset)[her_indexes]                 # 从tsamples和futureoffset家和后的64个index中取出her_index对应位置值,50
        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]   # [50,3]
        transitions['g'][her_indexes] = future_ag                              # 把transitions里面g部分换成future ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1) # 计算ag_next 和 g的奖励[64, 1]
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
                                                                     # ['obs',   'ag',    'g',   'actions','obs_next','ag_next',  'r']
                                                                     # [(64, 6), (64, 3), (64, 3), (64, 4), (64, 6), (64, 3), (64, 1)]

        return transitions
