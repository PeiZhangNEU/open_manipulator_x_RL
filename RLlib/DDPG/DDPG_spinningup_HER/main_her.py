import sys
import os
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径

from DDPGModel import *
import gym
import matplotlib.pyplot as plt
from RLlib.PPO.draw import Painter
import random
from copy import deepcopy

from gym_myrobot.envs.robot_reach import ReachEnv


def generate_goals(i, episode_cache, sample_num, sample_range=200):
    '''产生的是新的goals'''
    end = (i + sample_range) if (i + sample_range) < len(episode_cache) else len(episode_cache)
    trans_i_raw = episode_cache[i:end]
    if len(trans_i_raw) < sample_num:  # 如果i太靠后了，可能导致G_i长度小于samplenum=10,直接取
        trans_i = trans_i_raw
    else:
        trans_i = random.sample(trans_i_raw, sample_num)  # 如果i不靠后，Gi长度大于10，那就随机出10个
    
    # trans_i 是长度不定的列表，列表的每一项都是一个tansition=[observation, a, r, observation_]
    # observation 现在是 cat后的 [observation(pos+orn) + goal(pose)] 9维数据，我只要前三维度
    # her算法就是把 observation_ 作为新的 goal
    goals_i = [np.array(tran_i[3][:3]) for tran_i in trans_i]
    return goals_i

def evaluate(env, agent, episode):
    is_succes_times = 0
    obs = env.reset()
    observation = obs['observation']
    goal = obs['desired_goal']
    observation = np.concatenate((observation, goal))
    for _ in range(5):
        a = agent.get_action(observation, None, deterministic=True)
        obs_, r, d, _ = env.step(a)
        observation_ = obs_['observation']
        observation_ = np.concatenate((observation_, goal))
        if r == 1: # r==1就证明成功一次
            is_succes_times += 1
        obs = obs_
    succes_rate = is_succes_times/500
    return succes_rate

if __name__ == '__main__':
    env = ReachEnv(usegui=True, use_fixed_target=True)  # 假设目标goal是固定的，那么使用简单的her可否收敛？
    eval_env = deepcopy(env)
    obs_dim = 9 # 6obsdim + 3goaldim
    act_dim = 4 # 4个动作维度
    act_bound = [-env.action_space.high[0], env.action_space.high[0]]

    ddpg = DDPG(obs_dim, act_dim, act_bound)   # 初始化A算法，初始化Buffer

    Max_episodes = 100000
    Max_step = 500
    Her_sample_num = 10
    Update_times_N = 200
    batch_size = 256
    suc_rate_list = []
    
    for episode in range(Max_episodes):
        obs = env.reset()                 # obs 代表字典obs,reset时就采样了
        goal = obs['desired_goal']        # 得到采样得道的一个goal
        episode_cache = []

        # 获得目前episode的轨迹，并且存入buffer
        for t in range(Max_step): # t: [0-200)
            observation = obs['observation']  # 初始化得道初始化的obsevation 6维
            observation = np.concatenate((observation, goal))  # 输入是observation和goal的合体
            if episode > 50:
                a = ddpg.get_action(observation, ddpg.act_noise, deterministic=False)    # 加上噪声探索
            else:
                a = env.action_space.sample()
            obs_, r, d, _ = env.step(a)        # step可以自动计算r
            observation_ = obs_['observation']
            observation_ = np.concatenate((observation_, goal))

            episode_cache.append((observation, a, r, observation_))
            ddpg.replay_buffer.store(observation, a, r, observation_, d)
            obs = obs_
        # print(len(episode_cache))  # episod_cache的长度一直是200

        # Her：利用上一步得的目前episod轨迹，获得新的goals和新的r，存入buffer
        for i, transition in enumerate(episode_cache): # i: [0-200)
            # 从当前回合的 episode cache 生成 i 时刻的 G，也就是新的目标
            Gs_i = generate_goals(i, episode_cache, Her_sample_num)
            for g_ in Gs_i:
                o, a, o_ = transition[0], transition[1], transition[3]
                r_ = env.compute_reward(o[:3], g_, None)    # 计算采样出的o和新目标的奖励
                # 得到新的cocatnate的量
                o = np.concatenate((o[:6], g_))  # o原来是9维的，要去掉原来后面的goal，加新的goal
                o_ = np.concatenate((o_[:6], g_))
                # 存入buffer
                ddpg.replay_buffer.store(o, a, r, o_, False)
        
        # 从buffer中采样并进行训练
        if episode >= 50: # 随机采样50回合之后开始训练
            # 一次训练训练200次
            for _ in range(Update_times_N):
                # 采样minibatch B 从 buffer中
                mini_batch = ddpg.replay_buffer.sample_batch(batch_size=batch_size)
                ddpg.update(data=mini_batch)
        
        # # 完成训练之后进行一下测试
        # suc_rate = evaluate(eval_env, ddpg, episode)
        # suc_rate_list.append(suc_rate)
        # print('episode',episode, 'eval success rate',suc_rate)

        if episode % 20 == 0:
            """保存训练过程的模型"""
            torch.save(ddpg.ac.pi.state_dict(), 'DDPG_her_myopenmanipulator.pth')
            pass








        
        






            
            


            










        







