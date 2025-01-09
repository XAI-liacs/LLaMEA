import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(10 * dim, budget // 2)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # DE mutation factor
        self.CR = 0.9  # DE crossover rate
        self.initial_temp = 1.0
        self.final_temp = 0.01

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # DE Mutation and Crossover
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Simulated Annealing
                trial_fitness = func(trial)
                evaluations += 1
                if evaluations >= self.budget:
                    break
                
                temp = self.initial_temp * ((self.final_temp / self.initial_temp) ** (evaluations / self.budget))
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temp):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

        return best_solution