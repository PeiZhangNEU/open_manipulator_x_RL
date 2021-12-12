import numpy as np
# from gym.envs.robotics import rotations, robot_env, utils
import gym
import os
import math
import pybullet as p
from gym_myrobot.envs.robot_env import RobotEnv
from gym.utils import seeding
import pybullet_data
import random
import  time
from gym import spaces

'''
机械臂的reach到随机点的环境
输入动作为 arm的四个关节的delta量，范围在[-0.005,0.005]之间， 是个4 维数组
输出状态为字典形式的 observation{obs, achieved_goal, desired_goal}
                  其中，obs是目前的机械臂末端的 xyz以及euler的三元朝向，是个6维数组
升级版，可以在达到目标后锁死爪子
'''

def goal_distance(goal_a, goal_b):
    '''计算到目标位置的距离的差'''
    assert goal_a.shape == goal_b.shape
    # 计算两个坐标的差的2范数
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def get_target_pos():
    '''得到机械臂工作区域中的末端位置的xyz坐标，均匀采样'''
    flag = True
    while flag:
        x = 0.4 * random.random()
        y = 0.4 * (random.random() * 2.0 - 1.0)   # 0.4* ([0 ,2] - [1, 1]) = [-0.4, 0.4] 范围内的随机点
        z = 0.35 * random.random() + 0.05             # 由于生成0高度的点，夹子有宽度，所以中心点贴不到地面，所以需要向上一点移动。[0.05, 0.4]

        d_2 = x * x + y * y + z * z    # 点到球心的距离的平方为d_2

        # 如果距离平方小于半径平方，才返回这次生产的随机点，不然就再次生成一个随机点，这个叫做拒绝法，先在一个长方体里面生产随机点，再拒绝
        if d_2 < 0.4 *0.4:
            flag = False
        else:
            flag = True
    return x,y,z



class ReachEnv(gym.Env):
    '''
    继承gymEnv
    '''
    def __init__(self,
                 n_substeps=40,                 # 每次给的0.005的deltajoint的值，需要大概40/200=0.2s的时间才能执行完毕
                 distance_threshold=0.005,      # 设置终点和目标点之间距离的限制成功标准，因为方块比较小，经过实验，0.005是一个很好的限制范围，达到这个范围内可以很精确的重合 
                 reward_type='sparse', 
                 usegui=False,
                 usegripper=False,
                 use_fixed_target=False,
                 fixed_target=[0.1, 0.1, 0.1]):
        IS_USEGUI = usegui
        self.usegripper = usegripper
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.n_substeps=n_substeps
        self.n_actions=4
        self.targetUid=-1   # 每次reset都会把上次产生的目标点位删除，防止产生一堆点
        self.use_fixed_target = use_fixed_target   # 可以选择是否用固定点位，进行单点追踪
        self.fixed_target = fixed_target

        self._urdfRoot = pybullet_data.getDataPath()
        self.seed()
        # 是否进行渲染: 默认是false，但是可以创建环境的时候更改一下
        if IS_USEGUI:
            self.phisics = p.connect(p.GUI)
        else:
            self.phisics = p.connect(p.DIRECT)
        
        # 加载机器人，需要用到基环境
        self._robot = RobotEnv()
        self._timeStep = 1. / 200.
        action_dim = 4
        observation_dim = 9              # 这个设置为9是为了DRL的her算法设置的，因为输入是goal和observation的concate
        self._action_bound = 0.005
        self._observation_bound = np.inf
        action_high = np.array([self._action_bound] * action_dim)
        observation_high = np.array([self._observation_bound] * observation_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        # 重置环境
        self.reset()

    def compute_reward(self, achieved_goal, goal, info):
        '''计算奖励，有不稀疏和稀疏两种计算方法'''
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
    
    def step(self, action):
        '''执行动作并渲染'''
        action = np.clip(action, -self._action_bound, self._action_bound)
        self._set_action(action)

        for _ in range(self.n_substeps):
            '''仿真n个pybullet时间步数，因为怕执行不完'''
            p.stepSimulation()
        
        obs = self._get_obs()
        done = False
        info = {
            'is_success':self._is_success(obs['achieved_goal'], self.goal),
        }
        if self.usegripper:
            if info['is_success']:
                self._robot.operate_gripper(-0.01)

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    
    def reset(self):
        '''重置所有位置，包括机械臂以及目标位置，目标位置用一个没有实体的红色小方块来表示， 这个urdf可以加载进来我们机械臂末端小方块的模型，但是不实体化,
           在这里，目标位置的设置是随机点，需要大改！这里需要好好改一下才能用。
        '''
        rest_poses = [0.000, 0.000 ,0.000 ,0.000 ,0.010, 0.010]   # 前四个joint以及两个finger，要符合joint 的限定范围
        for i in range(6):
            p.resetJointState(self._robot.pandaUid, i, rest_poses[i], 0)  # 设置初始化的位置，以及速度
        p.setTimeStep(self._timeStep)

        # 目标小方块的pose, 也就是目标点的位置，和旋转方向，这个应该参照pick环境设置
        # Cube Pos: 是机械臂末端可以达到的一个工作空间，是一个1/4 球形域，而不是三个量都随机的一个正方体域，目标点要在这个球形域中产生！
        # target Position：
        if self.use_fixed_target:  # 如果使用单点追踪，那么目标点位每一轮都是固定的点
            xpos_target, ypos_target, zpos_target = self.fixed_target[0], self.fixed_target[1], self.fixed_target[2]
        else:
            xpos_target, ypos_target, zpos_target = get_target_pos()
        ang_target = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn_target = p.getQuaternionFromEuler([0, 0, ang_target])      

        # # 这里的if else 语句是为了每次reset都移除上次产生的目标点，不然会有好多个目标点一次次累加出现在屏幕上
        # self.targetUid = p.loadURDF("/home/zp/deeplearning/myrobot_plus/gym_myrobot/envs/cube_small_target_pick.urdf",    # 根据这个urdf来改我们之前的末端小方块的urdf
        #                                 [xpos_target, ypos_target, zpos_target],
        #                                  orn_target, useFixedBase=1)
        
        if self.targetUid == -1:  
            self.targetUid = p.loadURDF("./gym_myrobot/envs/cube_small_target_pick.urdf",    # 根据这个urdf来改我们之前的末端小方块的urdf
                                        [xpos_target, ypos_target, zpos_target],
                                         orn_target, useFixedBase=1)
        else:
            p.removeBody(self.targetUid)
            self.targetUid = p.loadURDF("./gym_myrobot/envs/cube_small_target_pick.urdf",    # 根据这个urdf来改我们之前的末端小方块的urdf
                                        [xpos_target, ypos_target, zpos_target],
                                         orn_target, useFixedBase=1)

        # 下面这个函数 setCollisionFilterPair 是取消碰撞检测，因为我们的机械臂设定了一个目标位置用小红点表示出来，但是这个是个虚拟的点，不需要发生碰撞！
        # body1，body2，body1的link序号，body2的link序号，0代表取消碰撞检测      避免了AB的对应link的碰撞！都取了倒数第一个link
        p.setCollisionFilterPair(self.targetUid, self._robot.pandaUid, -2, -1, 0)
        p.setCollisionFilterPair(self.targetUid, self._robot.pandaUid, -3, -1, 0)
        p.setCollisionFilterPair(self.targetUid, self._robot.pandaUid, -4, -1, 0)
        p.setCollisionFilterPair(self.targetUid, self._robot.pandaUid, -5, -1, 0)
        p.setCollisionFilterPair(self.targetUid, self._robot.pandaUid, -6, -1, 0)
        p.setCollisionFilterPair(self.targetUid, self._robot.pandaUid, -7, -1, 0)
        p.setCollisionFilterPair(self.targetUid, self._robot.pandaUid, -8, -1, 0)


        self.goal=np.array([xpos_target,ypos_target,zpos_target])

        p.setGravity(0, 0, 0)   # 真实的机械臂是不考虑重力的，因为就算给定关节控制量为0，机械臂也佁然不动！

        self._envStepCounter = 0
        obs = self._get_obs()
        self._observation = obs
        return self._observation
    

    def _set_action(self, action):
        '''利用基环境执行动作'''
        self._robot.applyAction(action)

    def _get_obs(self):
        '''获取机械臂的末端状态和朝向，规定字典形式的observation，由obs 和achieved goal与 desired goal组成'''
        end_pos_orn = np.array(self._robot.getObservation())
        # 前三个是xyz坐标
        end_pos = end_pos_orn[:3]
        # 后四个是orn四元量
        # 将得到的四元量的orn转换成Euler的叁元量，这个转换的函数对array生效，所以可以在真实环境中也使用！
        end_orn_quaternion = end_pos_orn[3:]
        end_orn = p.getEulerFromQuaternion(end_orn_quaternion)
        end_orn = np.array(end_orn)

        target_pos = np.array(p.getBasePositionAndOrientation(self.targetUid)[0])
        # obs 是两个array组成的列表，长度为0，1
        obs = [
            end_pos.flatten(),
            end_orn.flatten(),
        ]

        achieved_goal = end_pos.copy()

        # 把两个array组成的列表展开成一行array
        for i in range(1, len(obs)):
            end_pos = np.append(end_pos, obs[i])
        obs = end_pos.reshape(-1)

        self._observation = obs

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': target_pos.flatten(),
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _is_success(self, achieved_goal, desired_goal):
        '''根据末端位置和目标末端位置的距离判断是否成功'''
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

