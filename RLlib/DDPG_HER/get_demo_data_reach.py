#! /usr/bin/env python
import sys
import os
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径

import numpy as np
from mpi4py import MPI
#mpi4py是分布式计算的库   MPI:message passing interface 即消息传递接口
import random
from gym_myrobot.envs.robot_reach import ReachEnv
import pybullet as p

demo_num=1000
def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = 200
    return params

def launch():
    # create the ddpg_agent
    #创建环境，从参数文件里找
    env = ReachEnv(reward_type='sparse', usegui=False)
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment
    get_demo_data(env,env_params)

def get_demo_data(env,env_params):
    obs_total, ag_total, g_total, actions_total, info_total = [], [], [], [], []
    savetime = 0
    for epoch in range(2500):
        if savetime >= demo_num:
            break
        mb_obs, mb_ag, mb_g, mb_actions, mb_info = [], [], [], [], []
        # reset the rollouts
        ep_obs, ep_ag, ep_g, ep_actions, ep_info = [], [], [], [], []
        # reset the environment
        observation = env.reset()
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        # start to collect samples
        step_time = 0
        for t in range(int(env_params['max_timesteps'])):
            # print(t)
            step_time += 1
            # 由于我用的是 直接控制 joints 的值，所以，可以先用ik解出来达到目标点需要的前四个joint的值，作为desired joints值，然后再按照现在的joint值给偏差作为action
            target_joints = p.calculateInverseKinematics(env._robot.pandaUid, 6, g)
            target_joints = np.array(target_joints[:4])

            # get一下现在的joints 的值
            now_joints = np.array(p.getJointStates(env._robot.pandaUid, list(range(4))))[:, 0]

            # 通过偏差制造joint的控制action
            action = [target_joints[0]-now_joints[0], target_joints[1]-now_joints[1], target_joints[2]-now_joints[2], target_joints[3]-now_joints[3]]
            action = list(action)

            observation_new, _, _, info = env.step(action)
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']
            # append rollouts
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            ep_actions.append(action.copy())
            ep_info.append(info.copy())
            # re-assign the observation
            obs = obs_new
            ag = ag_new
        if info['is_success'] == 1.0:
            savetime += 1
            print("This is " + str(savetime) + " savetime ")
            ep_obs.append(obs.copy())  # obs 和 ag比另外两个多一行的原因出在这里
            ep_ag.append(ag.copy())    
            mb_obs.append(ep_obs)    
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
            mb_info.append(ep_info)
            # convert them into arrays
            obs_total.append(mb_obs)
            actions_total.append(mb_actions)
            g_total.append(mb_g)
            ag_total.append(mb_ag)
            info_total.append(mb_info)
    file = "myrobot_"+str(savetime)+"_reach_demo.npz"
    np.savez_compressed(file, acs=np.array(actions_total).squeeze(), obs=np.array(obs_total).squeeze(),
                        info=np.array(info_total).squeeze(), g=np.array(g_total).squeeze(),
                        ag=np.array(ag_total).squeeze())
if __name__ == '__main__':
    # get the params
    launch()
