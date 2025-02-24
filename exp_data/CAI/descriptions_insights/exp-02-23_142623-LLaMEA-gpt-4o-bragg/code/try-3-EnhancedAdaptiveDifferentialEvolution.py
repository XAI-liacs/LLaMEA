import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.F = 0.5  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.success_rate_threshold = 0.2
        self.adaptation_factor = 0.1

    def adapt_parameters(self, success_rate):
        if success_rate < self.success_rate_threshold:
            self.F = min(1.0, self.F + self.adaptation_factor)
            self.CR = max(0.0, self.CR - self.adaptation_factor)
        else:
            self.F = max(0.1, self.F - self.adaptation_factor)
            self.CR = min(1.0, self.CR + self.adaptation_factor)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        success_count = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    success_count += 1

                if evaluations >= self.budget:
                    break

            success_rate = success_count / self.population_size
            self.adapt_parameters(success_rate)
            success_count = 0

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

# Example usage:
# optimizer = EnhancedAdaptiveDifferentialEvolution(budget=10000, dim=5)
# best_solution, best_value = optimizer(func)