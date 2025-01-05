import numpy as np

class MultiSwarmQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.num_swarms = 5
        self.swarm_size = self.population_size // self.num_swarms
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.quantum_factor = 0.1

    def quantum_update(self, position, global_best, eval_count):
        delta = np.random.rand(self.dim)
        new_position = position + self.quantum_factor * (global_best - position) * delta
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        swarms = [np.random.rand(self.swarm_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
                  for _ in range(self.num_swarms)]
        swarm_best = [swarm[np.argmin([func(ind) for ind in swarm])] for swarm in swarms]
        global_best = min(swarm_best, key=lambda ind: func(ind))
        eval_count = self.num_swarms * self.swarm_size

        while eval_count < self.budget:
            for swarm_idx, swarm in enumerate(swarms):
                for i in range(self.swarm_size):
                    if eval_count >= self.budget:
                        break

                    a, b, c = swarm[np.random.choice(self.swarm_size, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < self.CR
                    trial = np.where(cross_points, mutant, swarm[i])
                    
                    trial = self.quantum_update(trial, global_best, eval_count)
                    trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                    trial_value = func(trial)
                    eval_count += 1

                    if trial_value < func(swarm[i]):
                        swarm[i] = trial

                swarm_best[swarm_idx] = min(swarm, key=lambda ind: func(ind))
                if func(swarm_best[swarm_idx]) < func(global_best):
                    global_best = swarm_best[swarm_idx]

        return global_best