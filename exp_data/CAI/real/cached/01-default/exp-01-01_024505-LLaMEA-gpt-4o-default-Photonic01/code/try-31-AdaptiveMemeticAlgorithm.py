import numpy as np

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(20, min(100, budget // 5))
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.crossover_rate = 0.8
        self.mutation_factor = 0.9
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.best_fitness:
            self.best_fitness = fitness[best_index]
            self.best_solution = self.population[best_index]
        return fitness

    def select_parents(self, fitness):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        return self.population[indices]

    def differential_evolution(self, parents, lb, ub):
        target, a, b, c = parents
        mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
        trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, target)
        return trial

    def simulated_annealing(self, solution, lb, ub):
        candidate = solution + np.random.normal(0, self.temperature, self.dim)
        candidate = np.clip(candidate, lb, ub)
        return candidate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += self.pop_size

            if evaluations >= self.budget:
                break

            new_population = []
            for _ in range(self.pop_size):
                parents = self.select_parents(fitness)
                trial = self.differential_evolution(parents, lb, ub)
                sa_trial = self.simulated_annealing(trial, lb, ub)
                trial_fitness = func(trial)
                sa_trial_fitness = func(sa_trial)
                if sa_trial_fitness < trial_fitness:
                    new_population.append(sa_trial)
                else:
                    new_population.append(trial)
                evaluations += 2  # Two additional evaluations

                if evaluations >= self.budget:
                    break

            self.population = np.array(new_population)
            self.temperature *= self.cooling_rate

        return self.best_solution, self.best_fitness