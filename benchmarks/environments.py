import gym

def run_env(env, steps=1000):
    env.reset()
    while steps > 0:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        steps -= 1
        if done:
            env.reset()

cartpole_env = gym.make('CartPole-v0')
# In [3]: %timeit run_env(cartpole_env)
# 8.68 ms ± 78.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)