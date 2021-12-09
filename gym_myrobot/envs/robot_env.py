import os, inspect
from numpy.core.records import recarray
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import  time

class RobotEnv:
    '''
    升级版程序，可以在达到目标之后锁定爪子
    '''
    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 4.8
        self.end_link = 6
        self.maxForce = 200.

        # 关节delta值的限定， 我认为可以更小
        self.joint_delta_limit_low = [-0.005, -0.005, -0.005, -0.005]
        self.joint_delta_limit_high = [0.005, 0.005, 0.005, 0.005]

        # 关节目标值的限定
        self.joints_limit_low = [-2.82743338823, -1.79070781255, -0.942477796077, -1.79070781255]
        self.joints_limit_high = [2.82743338823, 1.57079632679, 1.38230076758, 2.04203522483]

        # reset
        self.reset()

    def reset(self):
        '''初始化环境'''
        # 载入模型
        p.resetSimulation()
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0, 0, 0) # 重力应该设置为0，因为真实机械臂不受重力影响，即使不施加动作，机械臂也是保持静止的！
        p.setTimeStep(self.timeStep)
        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        # 绝对路径
        self.pandaUid = p.loadURDF("/home/pp/deeplearning/myrobot_plus/gym_myrobot/envs/open_manipulator.urdf", useFixedBase=True)  # fixedbase 是吧机械臂的底座固定住，不然每次仿真都会乱跑！
        # 重设所有joint的位置
        rest_poses = [0.000, 0.000 ,0.000 ,0.000 ,0.010, 0.010]   # 前四个joint以及两个finger，要符合joint 的限定范围
        for i in range(6):
            p.resetJointState(self.pandaUid,i, rest_poses[i])

        # 放一个桌子
        self.tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.63])

    def getActionDimension(self):
        '''这个环境只控制前4个关节的delta值，所以维度为4'''
        return 4

    def getObservationDimension(self):
        '''得到obser的维度'''
        return len(self.getObservation)
    
    def getObservation(self):
        '''获取环境的obs, 是末端笛卡尔坐标和朝向四元量的连接'''
        observation = []
        state = p.getLinkState(self.pandaUid, self.end_link)
        pos = state[4]     # 4比0得到的位置信息更加精确！
        pos = list(pos)
        orn = state[5]
        orn = list(orn)
        observation.extend(pos)
        observation.extend(orn)
        
        return observation

    def applyAction(self, action):
        '''把四个关节的delta值赋给关节'''
        action = np.clip(action,self.joint_delta_limit_low, self.joint_delta_limit_high)  # 把输出的动作裁剪到约定范围
        # action 
        # 对前置四个关节进行设定
        jointPoses_now = p.getJointStates(self.pandaUid, list(range(4)))
        jointPoses_now = np.array(jointPoses_now)[:, 0]
        jointPoses_target = jointPoses_now + action[:4]
        # 计算得到目标关节值之后，要对目标的值进行限定，保证不会损坏机械臂，限定值从urdf文件获取
        jointPoses_target = np.clip(jointPoses_target, self.joints_limit_low[:4], self.joints_limit_high[:4])
        p.setJointMotorControlArray(self.pandaUid, list(range(4)), p.POSITION_CONTROL, jointPoses_target)

    def operate_gripper(self, motor):
        '''单独控制爪子'''
        p.setJointMotorControlArray(self.pandaUid, [4,5], p.POSITION_CONTROL, [motor, motor])

    