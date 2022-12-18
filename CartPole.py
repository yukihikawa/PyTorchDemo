import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    #evn.step()方法有四个返回值
    observation, reward, done, info = env.step(action)
    print(observation)
env.close()