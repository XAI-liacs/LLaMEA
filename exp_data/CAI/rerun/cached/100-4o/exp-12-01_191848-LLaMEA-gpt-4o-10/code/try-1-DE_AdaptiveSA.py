import numpy as np

class DE_AdaptiveSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.init_temp = 100  # Initial temperature for simulated annealing
        self.cooling_rate = 0.95  # Cooling rate for simulated annealing

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, *self.bounds)
                
                # Crossover
                trial = np.copy(population[i])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial[cross_points] = mutant[cross_points]
                
                # Evaluate the trial individual
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Adaptive Simulated Annealing
                temperature = self.init_temp * np.power(self.cooling_rate, evaluations // self.pop_size)
                if np.exp((fitness[i] - trial_fitness) / temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Check budget
                if evaluations >= self.budget:
                    break

        return best_solution, best_fitness