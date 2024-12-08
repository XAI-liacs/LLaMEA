import numpy as np

class SynergisticFusionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.mutation_factor = 0.6
        self.crossover_probability = 0.85
        self.learning_rate = 0.05
        self.social_influence = 0.4
        self.population = None
        self.fitness = None

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                            (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def adaptive_differential_evolution(self, idx, func):
        indices = [i for i in range(self.population_size) if i != idx]
        np.random.shuffle(indices)
        a, b, c = indices[:3]
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        adaptive_factor = 0.3 + 0.7 * np.random.rand()
        mutant = self.population[idx] + adaptive_factor * (mutant - self.population[idx])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        
        trial = np.where(cross_points, mutant, self.population[idx])
        trial_fitness = func(trial)
        
        if trial_fitness < self.fitness[idx]:
            self.population[idx] = trial
            self.fitness[idx] = trial_fitness

    def swarm_influence(self, idx, func):
        best_idx = np.argmin(self.fitness)
        g_best = self.population[best_idx]
        direction_vector = self.social_influence * (g_best - self.population[idx])
        candidate = self.population[idx] + direction_vector
        candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
        candidate_fitness = func(candidate)
        if candidate_fitness < self.fitness[idx]:
            self.population[idx] = candidate
            self.fitness[idx] = candidate_fitness

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0
        self.evaluate_population(func)
        evaluations += self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                if np.random.rand() < 0.5:
                    self.adaptive_differential_evolution(i, func)
                else:
                    self.swarm_influence(i, func)
                evaluations += 1

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]