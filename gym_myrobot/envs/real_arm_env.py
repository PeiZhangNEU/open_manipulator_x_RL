#! /usr/bin/env python

'''
真实机械臂的任务环境，可以和仿真环境结构一致，但是其控制是使用ros控制。

动作包括了gripper！
其中状态为： obs为6维输入，xyz和欧拉朝向3元量
动作为 : action 为4维输入，四个关节的delta值
'''
from numpy.core.defchararray import join
import rospy
import numpy as np
# 导入写client需要用到的srv的包
from open_manipulator_msgs.srv import *
from rospy import client

# 导入写subscriber需要用到的msg的包
from open_manipulator_msgs.msg import KinematicsPose
from open_manipulator_msgs.msg import JointPosition
from sensor_msgs.msg import JointState
import os

# 导入random是为了产生目标位置
import random

# 导入pybullet是为了计算欧拉角
import pybullet as p
from gym import spaces
import gym

from gym.utils import seeding

def goal_distance(goal_a, goal_b):
    '''计算到目标位置的距离的差'''
    assert goal_a.shape == goal_b.shape
    # 计算两个坐标的差的2范数
    return np.linalg.norm(goal_a - goal_b, axis=-1)

# def get_target_pos():
#     '''得到机械臂工作区域中的末端位置的xyz坐标，均匀采样'''
#     flag = True
#     while flag:
#         x = 0.4 * random.random()
#         y = 0.4 * (random.random() * 2.0 - 1.0)   # 0.4* ([0 ,2] - [1, 1]) = [-0.4, 0.4] 范围内的随机点
#         z = 0.35 * random.random() + 0.05             # 由于生成0高度的点，夹子有宽度，所以中心点贴不到地面，所以需要向上一点移动。[0.05, 0.4]

#         d_2 = x * x + y * y + z * z    # 点到球心的距离的平方为d_2

#         # 如果距离平方小于半径平方，才返回这次生产的随机点，不然就再次生成一个随机点，这个叫做拒绝法，先在一个长方体里面生产随机点，再拒绝
#         if d_2 < 0.4 *0.4:
#             flag = False
#         else:
#             flag = True
#     return x,y,z


class RealarmEnv(gym.GoalEnv):
    '''这个环境专门为真实机械臂创建，有着可拓展性，以及符合任务环境的要求
       更新：12/13日
       目标点位置用和仿真环境一样的方法产生，先得到初始的gripper位置xyz，再在这个位置附近产生tar，随后拒绝掉工作区间外的点
    '''
    def __init__(self, 
                 distance_threshold=0.05,
                 reward_type='sparse',
                 use_fixed_target=False,
                 fixed_target=[0.1, 0.1, 0.1],
                 use_gripper=False,
                 target_range=0.15):
        # 启动ros
        self.start_ros()
        # 环境的参数
        self.use_gripper = use_gripper
        self.reward_type = reward_type
        self.use_fixed_target = use_fixed_target   # 可以选择是否用固定点位，进行单点追踪
        self.fixed_target = fixed_target
        self.distance_threshold = distance_threshold
        self.n_actions = 4
        action_dim = 4
        self._action_bound = 0.05          # action bound改成原来的10倍
        action_high = np.array([self._action_bound] * action_dim)
        # 动作范围
        self.action_space = spaces.Box(-action_high, action_high)
        # 关节的目标值限定
        self.joints_limit_low = [-2.82743338823, -1.79070781255, -0.942477796077, -1.79070781255]
        self.joints_limit_high = [2.82743338823, 1.57079632679, 1.38230076758, 2.04203522483]
        # 关节delta值的限定
        self.joint_delta_limit_low = [-0.005, -0.005, -0.005, -0.005]
        self.joint_delta_limit_high = [0.005, 0.005, 0.005, 0.005]
        self.obs_wait_time = 0.01
        self.excute_time = 0.01
        self.target_range = target_range
        self.seed()

        # 创建客户端对象，和服务端配套
        #  这是差量控制服务，用于执行差量动作
        self.client_set_joint_delta_goal = rospy.ServiceProxy('/goal_joint_space_path_from_present', SetJointPosition) # 这个本身就直接执行偏差控制量，不需要得到关节值了
        #  这是直接指定位置控制服务，用于reset时执行到初始位置0000
        self.client_set_joint_goal = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition)
        if self.use_gripper:
            # 这是控制爪子的client，可以不用，但是可以加上
            self.client_set_gripper_goal = rospy.ServiceProxy('/goal_tool_control', SetJointPosition)
        
        # 先把机械臂的各个joint初始化到目标位置
        self.reset_poses = [0.0, 0.0, 0.0, 0.0]
        # 机械臂执行初始joints直接到达
        self.excute_action_direct(self.reset_poses)
        rospy.sleep(3)  # 重置也需要时间啊，重置完之后等一会儿再getobs，再进行下一轮操作
        data_pose = rospy.wait_for_message('/gripper/kinematics_pose', KinematicsPose, timeout=None)
        end_pose = [data_pose.pose.position.x, data_pose.pose.position.y, data_pose.pose.position.z]
        end_pose = np.array(end_pose)
        self.initial_gripper_pos = end_pose

        # 重置环境
        self.reset()
        obs = self._get_obs()

        # 空间设置
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )

    def start_ros(self):
        '''启动ros node，创建类的同时必须首先创建node'''
        rospy.init_node('get_massages_and_control')

    def compute_reward(self, achieved_goal, goal, info):
        '''计算奖励，有稀疏和不稀疏两种算法'''
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    
    def operate_gripper(self, motor):
        '''单独为操作gripper写一个方法'''
        gripper_pose_req = SetJointPositionRequest()
        gripper_pose_req.planning_group = ''
        gripper_pose_req.path_time = np.float64(self.excute_time)
        # jointpose是一个子类, 根据srv的说明，一定要把各种格式转换为对应格式才行！比如要求的是float64，Python默认的是float32，所以需要更改一下！
        gripper_joint_pose = JointPosition()
        gripper_joint_pose.joint_name = ['gripper']
        gripper_joint_pose.position = [motor]
        gripper_pose_req.joint_position = gripper_joint_pose
        # # # 4.组织请求数据，并发送请求
        response_gripper = self.client_set_gripper_goal.call(gripper_pose_req)


    def excute_action_direct(self, action):
        '''直接给定各个关节的位置，机械臂直接执行到位置， 同理也要给set_joints加上限位！
           这个函数只是用于reset，不用于执行每次的delta 动作
        '''
        action = np.float64(action)
        set_joints = action
        # 对计算得到之后的目标joints值加入限定
        set_joints = np.clip(set_joints, self.joints_limit_low, self.joints_limit_high)
        # 先获取到要发送的请求的数据格式的模板, 比如我要做SetJointPosition,也就是'/goal_joint_space_path'服务。这个服务只能动arm的关节
        joint_pose_req = SetJointPositionRequest()
        joint_pose_req.planning_group = ''
        joint_pose_req.path_time = np.float64(3.0)
        # jointpose是一个子类, 根据srv的说明，一定要把各种格式转换为对应格式才行！比如要求的是float64，Python默认的是float32，所以需要更改一下！
        joint_pose = JointPosition()
        joint_pose.joint_name = ['joint1','joint2','joint3','joint4']
        joint_pose.position = set_joints[:4]
        joint_pose_req.joint_position = joint_pose
        # # 3.创建客户端对象
        # client_set_joint_goal = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition)
        # 4.组织请求数据，并发送请求
        response = self.client_set_joint_goal.call(joint_pose_req)

        if self.use_gripper:
            '''如果使用gripper，那么reset的时候就把爪子开到最大'''
            self.operate_gripper(0.01)

    def excute_action_delta(self, action):
        '''给定delta joint值，和目前机械臂的各个joint值，进行delta控制。
        经过实验，发现当机械臂执行到urdf文件中设置的joint limit值的时候，就会停止执行动作了，
        所以在加上偏差值之后，得到set_joints的时候，还需要对这个set_joints 的值进行裁剪，裁剪到urdf要求的限位值 
        '''
        action = np.float64(action)
        action = np.clip(action, self.joint_delta_limit_low, self.joint_delta_limit_high)  # 把输出的动作裁剪到约定范围
        set_joints = action
        joint_pose_req = SetJointPositionRequest()
        joint_pose_req.planning_group = ''
        joint_pose_req.path_time = np.float64(self.excute_time)
        # jointpose是一个子类, 根据srv的说明，一定要把各种格式转换为对应格式才行！比如要求的是float64，Python默认的是float32，所以需要更改一下！
        joint_pose = JointPosition()
        joint_pose.joint_name = ['joint1','joint2','joint3','joint4']
        joint_pose.position = set_joints[:4]
        joint_pose_req.joint_position = joint_pose
        # # 3.创建客户端对象
        # client_set_joint_goal = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition)
        # 4.组织请求数据，并发送请求
        response = self.client_set_joint_delta_goal.call(joint_pose_req)

    def reset(self):
        '''重置环境, 利用excute direct直接把机械臂复原到0000位置'''
        # 得到一个工作区域内的随机target位置
        if self.use_fixed_target:  # 如果使用单点追踪，那么目标点位每一轮都是固定的点
            xpos_target, ypos_target, zpos_target = self.fixed_target[0], self.fixed_target[1], self.fixed_target[2]
        else:
            # 换成目标点为target pose从机械臂初始点生成
            xpos_target, ypos_target, zpos_target = self._sample_goal()
        self.goal = np.array([xpos_target, ypos_target, zpos_target])
        # 设置机械臂初始joints位置
        reset_poses = [0.0, 0.0, 0.0, 0.0]
        # 机械臂执行初始joints直接到达
        self.excute_action_direct(reset_poses)
        rospy.sleep(3)  # 重置也需要时间啊，重置完之后等一会儿再getobs，再进行下一轮操作
        obs = self._get_obs()
        self._observation = obs
        return self._observation

    def _sample_goal(self):
        '''按照fetch env改的生成目标点的方法，这个方法生成的是立方体均匀采样的点
            这些点并不一定在工作区域内，所以可能会有无效点存在，所以需要剔除一些
        '''
        flag = True
        while flag:
            goal = self.initial_gripper_pos + self.np_random.uniform(
            -self.target_range, self.target_range, size=3
            )
            x, y, z = goal[0], goal[1], goal[2]
            d_2 = x * x + y * y + z * z    # 点到球心的距离的平方为d_2

            # 如果距离平方小于半径平方，才返回这次生产的随机点，不然就再次生成一个随机点，这个叫做拒绝法，先在一个长方体里面生产随机点，再拒绝
            if d_2 < 0.4 *0.4:
                flag = False
            else:
                flag = True
        return x,y,z

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''执行delta动作并返回observation'''
        self._set_action(action)
        # step执行动作和仿真环境不同，需要等待一定的时间，机械臂才能执行完毕这个动作，再去获取observation才准确
        rospy.sleep(self.obs_wait_time)
        obs = self._get_obs()
        done = False
        info = {
            'is_success':self._is_success(obs['achieved_goal'], self.goal),
        }
        # 在考虑爪子的情况下，如果达到目标，就锁死爪子
        if self.use_gripper:
            if info['is_success']:
                self.operate_gripper(-0.01)
                rospy.sleep(0.4)

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def _set_action(self, action):
        '''执行动作'''
        self.excute_action_delta(action)

    def _get_obs(self):
        '''获取机械臂的末端位置和朝向，并且得到字典形式的observation'''
        # 利用listenner获取得到现在的observation包括末端位置和朝向
        data_pose = rospy.wait_for_message('/gripper/kinematics_pose', KinematicsPose, timeout=None)
        end_pose = [data_pose.pose.position.x, data_pose.pose.position.y, data_pose.pose.position.z]
        end_pose = np.array(end_pose)
        end_orn_quaternion = [data_pose.pose.orientation.x, data_pose.pose.orientation.y, data_pose.pose.orientation.z, data_pose.pose.orientation.w]
        end_orn = p.getEulerFromQuaternion(end_orn_quaternion)
        end_orn = np.array(end_orn)
        # 从reset得到的目标位置获取这一轮操作的目标位置
        target_pose = self.goal

        # 建立obs
        obs = [end_pose.flatten(),
               end_orn.flatten()
               ]
        achieved_goal = end_pose.copy()
        for i in range(1, len(obs)):
            end_pose = np.append(end_pose, obs[i])
        obs = end_pose.reshape(-1)
        self._observation = obs

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': target_pose.flatten(),
        }

    def _is_success(self, achieved_goal, desired_goal):
        '''计算是否成功'''
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
    

        

        



