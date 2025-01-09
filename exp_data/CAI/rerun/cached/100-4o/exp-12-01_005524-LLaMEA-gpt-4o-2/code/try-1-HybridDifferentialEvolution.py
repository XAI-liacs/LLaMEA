import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0

    def __call__(self, func):
        # Evaluate the initial population
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.eval_count += 1
            if self.eval_count >= self.budget:
                return self._best_solution()
        
        # Main optimization loop
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation and Crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                random_factor = np.random.uniform(0.4, 0.6)
                mutant_vector = x1 + random_factor * (x2 - x3)  # Modified line with random scaling factor
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                # Adaptive Crossover
                crossover_prob = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover_prob):
                    crossover_prob[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover_prob, mutant_vector, self.population[i])
                
                # Local Search Mechanism
                if np.random.rand() < 0.5:
                    trial_vector += np.random.normal(0, 0.1, self.dim)
                    trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                # Selection
                trial_fitness = func(trial_vector)
                self.eval_count += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                if self.eval_count >= self.budget:
                    return self._best_solution()
        
        return self._best_solution()

    def _best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]