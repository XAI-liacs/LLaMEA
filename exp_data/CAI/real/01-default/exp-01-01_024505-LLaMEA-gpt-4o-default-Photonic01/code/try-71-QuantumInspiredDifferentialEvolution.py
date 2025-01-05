import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_agents = max(10, min(50, budget // 10))
        self.agents = None
        self.personal_best_positions = None
        self.personal_best_fitness = np.full(self.num_agents, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.f_diff = 0.8
        self.crossover_prob = 0.7
        self.interference_prob = 0.1

    def initialize_agents(self, lb, ub):
        self.agents = lb + (ub - lb) * np.random.rand(self.num_agents, self.dim)
        self.personal_best_positions = np.copy(self.agents)

    def evaluate_agents(self, func):
        fitness = np.array([func(agent) for agent in self.agents])
        for i, f in enumerate(fitness):
            if f < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = f
                self.personal_best_positions[i] = self.agents[i]
            if f < self.global_best_fitness:
                self.global_best_fitness = f
                self.global_best_position = self.agents[i]
        return fitness

    def mutate_and_crossover(self, lb, ub):
        new_agents = np.copy(self.agents)
        for i in range(self.num_agents):
            if np.random.rand() < self.interference_prob:
                interference_vector = lb + (ub - lb) * np.random.rand(self.dim)
                new_agents[i] = np.mean([self.agents[i], interference_vector], axis=0)
            else:
                indices = np.random.choice(list(range(i)) + list(range(i + 1, self.num_agents)), 3, replace=False)
                a, b, c = self.agents[indices]
                mutant_vector = a + self.f_diff * (b - c)
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_prob or j == j_rand:
                        new_agents[i, j] = mutant_vector[j]
            new_agents[i] = np.clip(new_agents[i], lb, ub)
        self.agents = new_agents

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_agents(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_agents(func)
            evaluations += self.num_agents

            if evaluations >= self.budget:
                break

            self.mutate_and_crossover(lb, ub)

        return self.global_best_position, self.global_best_fitness