import numpy as np

class EMAO:
    def __init__(self, budget, dim, num_agents=10, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.num_agents = num_agents
        self.max_iter = max_iter

    def __call__(self, func):
        agents = np.random.uniform(-5.0, 5.0, size=(self.num_agents, self.dim))
        for _ in range(self.max_iter):
            for i in range(self.num_agents):
                if self.budget <= 0:
                    break
                new_position = agents[i] + np.random.uniform(-0.1, 0.1, size=self.dim)
                if func(new_position) < func(agents[i]):
                    agents[i] = new_position
                self.budget -= 1
            if _ % 10 == 0:  # Share information every 10 iterations
                best_agent = agents[np.argmin([func(agent) for agent in agents])]
                agents = [best_agent + np.random.uniform(-0.5, 0.5, size=self.dim) for _ in range(self.num_agents)]
        return agents[np.argmin([func(agent) for agent in agents])]