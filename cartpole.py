import gym
import numpy as np

class CartpoleAgent():
    MAX_STEPS = 1000

    def randObs():
        return np.random.rand(4) * 2 - 1

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.w = np.zeros(4)

    def train(self, n_iter=1000):
        self.w = CartpoleAgent.randObs()

    def test(self, n_iter=1000):
        """
        Returns fraction of good results.
        """
        n_success = 0
        for i in range(n_iter):
            obs = self.env.reset()
            total_reward = 0
            for j in range(self.MAX_STEPS):
                action = self.getAction(obs)
                obs, reward, done, info = self.env.step(action)
                if done:
                    break
                else:
                    total_reward += reward
                    if total_reward == 200:
                        n_success += 1
                        break
        return n_success / n_iter

    def getAction(self, obs, w=None):
        if w == None: w = self.w
        if np.dot(w, obs) > 0: return 1
        else: return 0

class RandomCartpoleAgent(CartpoleAgent):
    def train(self, n_iter=1000):
        w_opt, max_total_reward = self.w, 0
        for i in range(n_iter):
            w = CartpoleAgent.randObs()
            total_reward = 0

            obs = self.env.reset()
            for j in range(self.MAX_STEPS):
                action = self.getAction(obs, w)
                obs, reward, done, info = self.env.step(action)
                if done: break
                else: total_reward += reward
            if total_reward > max_total_reward:
                w_opt, max_total_reward = w, total_reward
        self.w = w_opt

class HillCartpoleAgent(CartpoleAgent):
    def train(self, alpha, n_iter=10000):
        w_opt, max_total_reward = CartpoleAgent.randObs(), 0
        for i in range(n_iter):
            w = w_opt + alpha * CartpoleAgent.randObs()
            total_reward = 0

            obs = self.env.reset()
            for j in range(self.MAX_STEPS):
                action = self.getAction(obs, w)
                obs, reward, done, info = self.env.step(action)
                if done: break
                else: total_reward += reward
            if total_reward > max_total_reward:
                w_opt, max_total_reward = w, total_reward
        self.w = w_opt

N_ITER = 10
if __name__ == '__main__':
    for i in range(N_ITER):
        rand_agent = RandomCartpoleAgent()
        rand_agent.train()
        percent_success = rand_agent.test() * 100
        print('Successes: ' + str(percent_success) + '%')
    for i in range(N_ITER):
        hill_agent = HillCartpoleAgent()
        hill_agent.train(0.001)
        percent_success = hill_agent.test() * 100
        print('Successes: ' + str(percent_success) + '%')


