import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # standard population size for DE
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf

    def initialize_population(self):
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.pop_size, self.dim)
        )

    def evaluate_population(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        return fitness

    def select_parents(self, index):
        indices = list(range(self.pop_size))
        indices.remove(index)
        np.random.shuffle(indices)
        return indices[:3]

    def mutate(self, target_idx):
        a, b, c = self.select_parents(target_idx)
        mutant_vector = (
            self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        )
        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
        return mutant_vector

    def crossover(self, target_idx, mutant_vector):
        trial_vector = np.copy(self.population[target_idx])
        crossover_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_points):
            crossover_points[np.random.randint(0, self.dim)] = True
        trial_vector[crossover_points] = mutant_vector[crossover_points]
        return trial_vector

    def adapt_parameters(self, fitness_history):
        if len(fitness_history) > 0:
            success_rate = len([f for f in fitness_history if f < 0]) / len(fitness_history)
            self.mutation_factor = 0.5 + 0.5 * success_rate
            self.crossover_rate = 0.6 + 0.3 * success_rate

    def __call__(self, func):
        self.initialize_population()
        num_evaluations = 0
        fitness_history = []
        
        while num_evaluations < self.budget:
            fitness = self.evaluate_population(func)
            for i in range(self.pop_size):
                if num_evaluations >= self.budget:
                    break

                mutant_vector = self.mutate(i)
                trial_vector = self.crossover(i, mutant_vector)
                trial_fitness = func(trial_vector)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness
                    fitness_history.append(trial_fitness - fitness[i])
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial_vector

            self.adapt_parameters(fitness_history)
            fitness_history.clear()

        return self.best_solution, self.best_fitness