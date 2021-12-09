# open_manipulator_x_RL
## 1. Introduction
使用了Openmanipulator-X这一开源机械臂作为控制对象，进行RL在仿真环境中的训练以及模型在实际环境中的使用。
参考网址：https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/
操作系统：Ubuntu20.04
Ros版本：noetic
具体安装ros以及机械臂sdk环境请参考官网链接
RL环境：python3.6，首先安装stable-baseline3环境，然后安装rospackage，再安装pybullet，再安装netifaces即可。
stable-baseline3安装参考https://link.zhihu.com/?target=https%3A//stable-baselines3.readthedocs.io/en/master/

## 2. Structure
open_manipulator_X_Rl
--gym_myrobot
  --envs
      __init__.py
      meshes
      --xxx.stl
      --xxx.stl
      --...
      cube_small_target.urdf
      open_manipulator.urdf
      robot_env.py
      robot_reach.py
      real_arm_env.py
  --__init__.py
--sim_real_test.ipynb
--use_ppo_fixedtarget_train.ipynb

## 3.内容介绍
主要包含环境和训练代码两部分
### 3.1 环境
环境包括仿真环境和实际环境
仿真环境：
使用pybullet构建，主要包含robot_env.py 和 robot_reach.py.其中robot_env.py是使用gymenv的基环境，
仿真环境可以设定固定点位为目标位置作为一个固定位置追踪的简单RL gym环境，也可以设定随机位置为目标位置作为一个 RL goal gym环境。
环境主体的设计是按照 goal gym环境的标准设计的，observation由字典形式组成。

实际环境：
实际环境结合了Openmanipulator-x的官方ros server list构建。使用client发送请求，使用rospy.wait接受目前的状态
实际环境对应着 real_arm_env.py 文件
首先在终端执行 `roslaunch open_manipulator_controller open_manipulator_controller.launch`，然后就可以像生成一个gym环境一样生成真实机械臂的操作环境了

### 3.2 Agent
目前仅仅使用了PPO算法结合固定目标位置对 普通的gym环境进行了训练与测试。 但本程序的环境本身是task env，可以结合HER buffer 以及其他RL Agent做出更加复杂的任务（本环境已经有了随机点跟踪任务）


### 3.3 Agent的输入和输出
obseration dim = 6 包括末端位置的笛卡尔坐标+朝向三元数组
action dim = 4 包括四个关节的delta控制量
控制目标，通过输出每个时刻四个关节的delta控制量，控制机械臂末端位置达到目标位置并且合上机械爪。

## 4.本程序仅供参考

## 5.后续更新
后续将更新其他RLAgent算法在arm上的应用，以及对HER算法的实现
