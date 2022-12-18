import gym
import numpy as np

env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))

class BespokeAgent: #简单的模拟智能体，不能真正
    def __init__(self, env):
        pass

    def decide(self, observation): #决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action # 返回动作

    def learn(self, *args):
        pass

agent = BespokeAgent(env)

def play_montecarlo(env, agent, render = False, train = False):
    episode_reward = 0 #回合总奖励，初始化为0
    observation = env.reset() #重置游戏环境，开始新回合
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action) # 执行动作
        episode_reward += reward
        if train: #是否训练智能体
            agent.learn(observation, action, reward, done)
        if done: #回合结束
            break
        observation = next_observation
    return episode_reward

env.seed(0) # 随机种子，为了复现结果，可删去
episode_reward = play_montecarlo(env, agent, render=True)
print('回合奖励 = {}'.format(episode_reward))
env.close()

episode_reward = [play_montecarlo(env, agent) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_reward)))