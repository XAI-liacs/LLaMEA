import numpy as np

class SwarmDrivenQuantumEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.quantum_superposition_rate = 0.6
        self.entanglement_factor = 0.5
        self.history = []

    def quantum_superposition(self, swarm):
        superposed = np.random.rand(self.swarm_size, self.dim)
        mask = np.random.rand(self.swarm_size, self.dim) < self.quantum_superposition_rate
        return np.where(mask, superposed, swarm)

    def swarm_intelligence(self, swarm, best):
        global_influence = np.random.rand(self.dim) * (best - swarm)
        local_influence = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        new_positions = swarm + self.entanglement_factor * (global_influence + local_influence)
        return np.clip(new_positions, self.bounds.lb, self.bounds.ub)

    def __call__(self, func):
        self.bounds = func.bounds
        swarm = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.swarm_size, self.dim))
        scores = np.array([func(ind) for ind in swarm])
        best_idx = np.argmin(scores)
        best_solution = swarm[best_idx]

        self.history.extend(scores)

        evaluations = self.swarm_size
        while evaluations < self.budget:
            superposed_swarm = self.quantum_superposition(swarm)
            new_swarm = self.swarm_intelligence(superposed_swarm, best_solution)

            for i in range(self.swarm_size):
                new_score = func(new_swarm[i])
                evaluations += 1

                if new_score < scores[i]:
                    scores[i] = new_score
                    swarm[i] = new_swarm[i]

                if new_score < scores[best_idx]:
                    best_idx = i
                    best_solution = new_swarm[i]

            self.history.extend(scores)

        return best_solution, scores[best_idx], self.history