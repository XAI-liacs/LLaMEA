import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.5
        self.population = None

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def adapt_parameters(self, generation):
        # Linearly varying mutation and crossover to improve convergence
        self.mutation_factor = 0.5 + (0.9 - 0.5) * (generation / (self.budget // self.pop_size))
        self.crossover_rate = 0.9 - 0.5 * (generation / (self.budget // self.pop_size))

    def select_parents(self, idx):
        candidates = list(range(self.pop_size))
        candidates.remove(idx)
        return np.random.choice(candidates, 3, replace=False)

    def mutate(self, target_idx, parents):
        a, b, c = self.population[parents]
        mutant_vector = a + self.mutation_factor * (b - c)
        return np.clip(mutant_vector, -5.0, 5.0)

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        return np.where(crossover_mask, mutant_vector, target_vector)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        evaluations = 0
        generation = 0

        while evaluations < self.budget:
            self.adapt_parameters(generation)
            new_population = []

            for i in range(self.pop_size):
                parents = self.select_parents(i)
                mutant_vector = self.mutate(i, parents)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                
                f_trial = func(trial_vector)
                evaluations += 1
                
                if f_trial < func(self.population[i]):
                    new_population.append(trial_vector)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial_vector
                else:
                    new_population.append(self.population[i])

                if evaluations >= self.budget:
                    break

            self.population = np.array(new_population)
            generation += 1

        return self.f_opt, self.x_opt