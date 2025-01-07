import numpy as np

class QuantumWalkInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.scale_factor = 0.5
        self.crossover_prob = 0.7

    def quantum_walk(self, position, global_best, step_size):
        # Quantum walk inspired transition
        q_step = np.random.normal(0, step_size, size=self.dim)
        new_position = position + q_step * (global_best - position)
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        global_best = pop[np.argmin(fitness)]
        global_best_value = fitness.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutate using Differential Evolution strategy
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[indices]
                mutant = np.clip(a + self.scale_factor * (b - c), bounds[:, 0], bounds[:, 1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Quantum walk update
                trial = self.quantum_walk(trial, global_best, step_size=0.1)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                trial_value = func(trial)
                eval_count += 1
                if trial_value < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if eval_count >= self.budget:
                    break

        return global_best