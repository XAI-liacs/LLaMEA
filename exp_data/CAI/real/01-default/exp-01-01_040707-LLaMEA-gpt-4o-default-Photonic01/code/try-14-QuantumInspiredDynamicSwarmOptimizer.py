import numpy as np

class QuantumInspiredDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5
        self.beta = 0.5
        self.q_probability = 0.1
        self.topology_update_freq = 10
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) - 0.5
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.topology = np.random.choice(self.population_size, (self.population_size, 3), replace=False)

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest[i] = self.position[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
        return scores

    def update_velocity_position(self, iteration, max_iterations, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            quantum_jumps = np.random.rand(self.dim) < self.q_probability
            if quantum_jumps.any():
                self.position[i][quantum_jumps] = lb[quantum_jumps] + (ub[quantum_jumps] - lb[quantum_jumps]) * np.random.rand(np.sum(quantum_jumps))
            else:
                local_best = self.pbest[self.topology[i]].min(axis=0)
                cognitive = self.alpha * r1[i] * (self.pbest[i] - self.position[i])
                social = self.beta * r2[i] * (local_best - self.position[i])
                self.velocity[i] = cognitive + social
                self.position[i] += self.velocity[i]
                self.position[i] = np.clip(self.position[i], lb, ub)

    def update_topology(self):
        self.topology = np.random.choice(self.population_size, (self.population_size, 3), replace=False)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position(iteration, max_iterations, func.bounds)
            if iteration % self.topology_update_freq == 0:
                self.update_topology()
            iteration += 1
        return self.gbest, self.gbest_score