import numpy as np

class ChaosDrivenAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.beta = 0.05  # Chaotic parameter

    def chaotic_sequence(self, size):
        # Generate a chaotic sequence using the logistic map for diversity
        x = np.random.rand()
        sequence = []
        for _ in range(size):
            x = 4 * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)

    def adapt_parameters(self, eval_count):
        # Adapt mutation and crossover rates
        lambda_factor = eval_count / self.budget
        self.mutation_factor = 0.5 * (1 + lambda_factor)
        self.crossover_rate = 0.9 * (1 - lambda_factor)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        best_idx = np.argmin([func(ind) for ind in pop])
        best = pop[best_idx]

        eval_count = self.population_size
        chaotic_sequence = self.chaotic_sequence(self.population_size * self.dim).reshape(self.population_size, self.dim)

        while eval_count < self.budget:
            self.adapt_parameters(eval_count)
            new_pop = np.empty_like(pop)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[indices]

                mutant = x1 + self.mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                trial = np.clip(trial + self.beta * chaotic_sequence[i], bounds[:, 0], bounds[:, 1])
                
                if func(trial) < func(pop[i]):
                    new_pop[i] = trial
                    if func(trial) < func(best):
                        best = trial
                else:
                    new_pop[i] = pop[i]
                
                eval_count += 1
                if eval_count >= self.budget:
                    break

            pop = new_pop

        return best