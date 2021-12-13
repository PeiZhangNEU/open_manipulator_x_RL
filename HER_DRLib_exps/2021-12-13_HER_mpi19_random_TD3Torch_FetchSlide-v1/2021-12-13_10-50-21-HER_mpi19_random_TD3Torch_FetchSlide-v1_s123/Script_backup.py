import sys
import os
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径

import numpy as np
import gym
import os, sys
from arguments import get_args
import torch
from mpi4py import MPI
from subprocess import CalledProcessError

import time
from spinup_utils.logx import setup_logger_kwargs, colorize
from spinup_utils.logx import EpochLogger
from spinup_utils.print_logger import Logger

from gym_myrobot.envs.robot_reach import ReachEnv
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
but I ignore it~

"""


def obs2state(obs, key_list=['observation', 'desired_goal']):
    if type(obs) == dict:
        s = np.concatenate(([obs[key] for key in key_list]
                            ))
    elif type(obs) == np.ndarray:
        s = obs[:]
    else:
        s = obs[:]
    return s


def trainer(net, env, args):
    # logger
    exp_name = args.exp_name+'_'+args.RL_name+'_'+args.env_name
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name,
                                        seed=args.seed,
                                        output_dir=args.output_dir + "/")
    logger = EpochLogger(**logger_kwargs)
    sys.stdout = Logger(logger_kwargs["output_dir"] + "print.log",
                        sys.stdout)
    logger.save_config(locals(), __file__)
    # start running
    start_time = time.time()
    for i in range(args.n_epochs):
        for c in range(args.n_cycles):
            obs = env.reset()
            episode_trans = []
            s = obs2state(obs)
            ep_reward = 0
            real_ep_reward = 0
            episode_time = time.time()

            success = []
            for j in range(args.n_steps):
                a = net.get_action(s, noise_scale=args.noise_ps)
                # a = net.get_action(s)
                # a = noise.add_noise(a)

                if np.random.random() < args.random_eps:
                    a = np.random.uniform(low=-net.a_bound,
                                          high=net.a_bound,
                                          size=net.act_dim)
                a = np.clip(a, -net.a_bound, net.a_bound)
                # ensure the gripper close!
                try:
                    obs_next, r, done, info = env.step(a)
                    success.append(info["is_success"])
                except Exception as e:
                    success.append(int(done))
                s_ = obs2state(obs_next)

                # visualization
                if args.render and i % 3 == 0 and c % 20 == 0:
                    env.render()

                # 防止gym中的最大step会返回done=True
                done = False if j == args.n_steps - 1 else done

                if not args.her:
                    net.store_transition((s, a, r, s_, done))

                episode_trans.append([obs, a, r, obs_next, done, info])
                s = s_
                obs = obs_next
                ep_reward += r
                real_ep_reward += r
            if args.her:
                net.save_episode(episode_trans=episode_trans,
                                 reward_func=env.compute_reward)
            logger.store(EpRet=ep_reward)
            logger.store(EpRealRet=real_ep_reward)

            for _ in range(40):
                outs = net.learn(args.batch_size,
                                 args.base_lr,
                                 args.base_lr * 2,
                                 )
                if outs[1] is not None:
                    logger.store(Q1=outs[1])
                    logger.store(Q2=outs[2])
            if 0.0 < sum(success) < args.n_steps:
                print("epoch:", i,
                      "\tep:", c,
                      "\tep_rew:", ep_reward,
                      "\ttime:", np.round(time.time()-episode_time, 3),
                      '\tdone:', sum(success))

        # # 存储每一个epoch的模型actor
        # net.save_simple_network(logger_kwargs["output_dir"])

        test_ep_reward, logger = net.test_agent(args=args,
                                                env=env,
                                                n=10,
                                                logger=logger,
                                                obs2state=obs2state,
                                                )
        logger.store(TestEpRet=test_ep_reward)

        logger.log_tabular('Epoch', i)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpRealRet', average_only=True)
        logger.log_tabular('TestEpRet', average_only=True)

        logger.log_tabular('Q1', with_min_and_max=True)
        logger.log_tabular('Q2', average_only=True)

        logger.log_tabular('TestSuccess', average_only=True)

        logger.log_tabular('TotalEnvInteracts', i * args.n_cycles * args.n_steps + c * args.n_steps + j + 1)
        logger.log_tabular('TotalTime', time.time() - start_time)
        logger.dump_tabular()

    print(colorize("the experience %s is end" % logger.output_file.name,
                   'green', bold=True))
    net.save_simple_network(logger_kwargs["output_dir"])


def launch(net, args):
    # env = gym.make(args.env_name)
    env = ReachEnv(usegui=True)  
    env.seed(args.seed)
    np.random.seed(args.seed)

    try:
        s_dim = env.observation_space.shape[0]
    except:
        s_dim = env.observation_space.spaces['observation'].shape[0] + \
                env.observation_space.spaces['desired_goal'].shape[0]

    act_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    """        
        torch1.17.1，gpu_id: 1 device: cuda:0，用的是物理上的0卡；
        cuda的序号仍然是按照物理序号；
        torch1.3.1，gpu_id: 1 device: cuda:0，用的是物理上的1卡，
        torch1.3.1，gpu_id: 1 device: cuda:1，报错：invalid device ordinal；
        torch1.3.1，gpu_id: 1,3 device: cuda:1，用的是物理上的3卡，
        有点类似于指定GPU-ID后，cuda会重新排序。        
    """

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id != -1 else 'cpu')
    print("gpu_id:", args.gpu_id,
          "device:", device)

    net = net(act_dim=act_dim,
              obs_dim=s_dim,
              a_bound=a_bound,
              per_flag=args.per,
              her_flag=args.her,
              action_l2=args.action_l2,
              state_norm=args.state_norm,
              gamma=args.gamma,
              sess_opt=args.sess_opt,
              seed=args.seed,
              clip_return=args.clip_return,
              device=device,
              )
    # restore_path = 'HER_DRLib_exps/2021-02-22_HER_TD3Torch_FetchPush-v1/2021-02-22_14-46-52-HER_TD3Torch_FetchPush-v1_s123/actor.pth'
    # net.load_simple_network(restore_path)
    trainer(net, env, args)


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    
    # get the params
    args = get_args()
    from algos.tf1.td3_sp.TD3_per_her import TD3
    from algos.tf1.ddpg_sp.DDPG_per_her import DDPG
    from algos.tf1.sac_sp.SAC_per_her import SAC
    from algos.tf1.sac_auto.sac_auto_per_her import SAC_AUTO

    from algos.pytorch.td3_sp.td3_per_her import TD3Torch
    from algos.pytorch.ddpg_sp.ddpg_per_her import DDPGTorch
    from algos.pytorch.sac_sp.sac_per_her import SACTorch

    RL_list = [TD3, DDPG, SAC, SAC_AUTO, TD3Torch, DDPGTorch, SACTorch]

    [launch(net=net, args=args) for net in RL_list if net.__name__ == args.RL_name]
