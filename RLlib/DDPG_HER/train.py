#! /usr/bin/env python
import sys
import os
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径


import numpy as np
import os
from arguments import get_args, Args
from mpi4py import MPI
#mpi4py是分布式计算的库
from ddpg_agent import ddpg_agent
import random
import torch
from gym_myrobot.envs.robot_reach import ReachEnv
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""
def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = 200
    return params


def launch(args):
    # create the ddpg_agent
    #创建环境，从参数文件里找
    env = ReachEnv(usegui=True)
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    print(env_params)
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()
    ddpg_trainer.plot_success_rate()


if __name__ == '__main__':
    # take the configuration for the HER
  #  os.environ['OMP_NUM_THREADS'] = '10'
    #使用python获得系统的信息时，使用os.environ()
    '''最近在使用pytorch中遇到的问题，python默认开启了多线程，导致一个程序占据占据了服务器的大半资源，可通过export
    OMP_NUM_THREADS = 1，将当前终端限制只使用单线程，该方法对pycaffe也有效。
    '''
    #os.environ['MKL_NUM_THREADS'] = '1'
    #设置mkl的线程，mkl是intel的数学库？
   # os.environ['IN_MPI'] = '10'
    # get the params
    args = Args()
    launch(args)
