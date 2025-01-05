import numpy as np

class AdaptiveDifferentialEvolutionWithChaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.scale_factor = 0.5
        self.crossover_rate = 0.9
        self.chaos_sequence = self.generate_chaos_sequence(self.population_size)

    def generate_chaos_sequence(self, size):
        x = np.zeros(size)
        x[0] = np.random.rand()
        for i in range(1, size):
            x[i] = 4 * x[i-1] * (1 - x[i-1])  # Logistic map
        return x

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(individual) for individual in self.population])
        for i, f in enumerate(fitness):
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_solution = self.population[i]
        return fitness

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = (self.population[a] +
                         self.scale_factor * (self.population[b] - self.population[c]))
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = (np.random.rand(self.dim) < self.crossover_rate)
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def adaptive_population_size(self, evaluations):
        return min(self.population_size, int(self.population_size * (1 - evaluations / self.budget)))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            current_pop_size = self.adaptive_population_size(evaluations)
            fitness = self.evaluate_population(func)
            evaluations += current_pop_size

            if evaluations >= self.budget:
                break

            for i in range(current_pop_size):
                mutant_vector = self.mutate(i)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                trial_vector = np.clip(trial_vector, lb, ub)

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector

        return self.best_solution, self.best_fitness