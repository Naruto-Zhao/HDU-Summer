import gym

# 创建一个环境
env = gym.make('CartPole-v0')
for i_episode in range(20):
    # 初始化一个环境
    observation = env.reset()
    for t in range(100):
        # 环境改变
        env.render()
        print(observation)
        # 进行一个动作
        action = env.action_space.sample()
        # 返回四个值：观察、奖励、完成情况、调试信息
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()