import numpy as np

class QE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.swarm = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_scores = None
        self.gbest_position = None
        self.gbest_score = float('inf')
        self.evaluations = 0

    def initialize_swarm(self, lb, ub):
        self.swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.pbest_positions = self.swarm.copy()
        self.pbest_scores = np.array([self.evaluate(ind) for ind in self.swarm])
        self.update_gbest()

    def evaluate(self, solution):
        return self.func(solution)

    def update_gbest(self):
        best_idx = np.argmin(self.pbest_scores)
        if self.pbest_scores[best_idx] < self.gbest_score:
            self.gbest_position = self.pbest_positions[best_idx].copy()
            self.gbest_score = self.pbest_scores[best_idx]

    def quantum_entanglement(self):
        for i in range(self.swarm_size):
            entanglement_effect = np.random.uniform(-0.5, 0.5, self.dim)
            self.swarm[i] += entanglement_effect * np.abs(self.swarm[i] - self.gbest_position)
            self.swarm[i] = np.clip(self.swarm[i], self.func.bounds.lb, self.func.bounds.ub)

    def update_velocities_and_positions(self, lb, ub):
        for i in range(self.swarm_size):
            inertia = self.inertia_weight * self.velocities[i]
            cognitive = self.cognitive_component * np.random.rand(self.dim) * (self.pbest_positions[i] - self.swarm[i])
            social = self.social_component * np.random.rand(self.dim) * (self.gbest_position - self.swarm[i])

            self.velocities[i] = inertia + cognitive + social
            self.swarm[i] += self.velocities[i]
            self.swarm[i] = np.clip(self.swarm[i], lb, ub)

            new_score = self.evaluate(self.swarm[i])
            if new_score < self.pbest_scores[i]:
                self.pbest_positions[i] = self.swarm[i].copy()
                self.pbest_scores[i] = new_score

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_swarm(lb, ub)

        while self.evaluations < self.budget:
            self.update_velocities_and_positions(lb, ub)
            self.update_gbest()
            self.quantum_entanglement()
            self.evaluations += self.swarm_size

        return {'solution': self.gbest_position, 'fitness': self.gbest_score}