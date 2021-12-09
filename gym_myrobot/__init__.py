from gym.envs.registration import register

register(
    id='MyRobot-v0',
    entry_point='gym_myrobot.envs:RobotEnv',
)