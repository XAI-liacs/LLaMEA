import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(100, self.budget // 2)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.eval_counter = 0

    def evaluate(self, func, individual):
        if self.eval_counter < self.budget:
            self.eval_counter += 1
            return func(individual)
        else:
            return np.inf

    def mutate(self, target_idx):
        indices = np.random.choice(np.delete(np.arange(self.pop_size), target_idx), 3, replace=False)
        donor_vector = self.population[indices[0]] + self.mutation_factor * (self.population[indices[1]] - self.population[indices[2]])
        return np.clip(donor_vector, self.lower_bound, self.upper_bound)

    def crossover(self, target, donor):
        return np.array([donor[i] if np.random.rand() < self.crossover_probability else target[i] for i in range(self.dim)])

    def select(self, target_idx, trial_vector, func):
        trial_fitness = self.evaluate(func, trial_vector)
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial_vector
            self.fitness[target_idx] = trial_fitness

    def __call__(self, func):
        if self.eval_counter >= self.budget:
            return np.min(self.fitness), self.population[np.argmin(self.fitness)]
        
        # Initialize fitness
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])

        while self.eval_counter < self.budget:
            for i in range(self.pop_size):
                donor = self.mutate(i)
                trial = self.crossover(self.population[i], donor)
                self.select(i, trial, func)
            
            # Adaptive tuning of parameters
            if self.eval_counter % (self.pop_size * 10) == 0:
                self.mutation_factor = np.random.uniform(0.5, 1.0)
                self.crossover_probability = np.random.uniform(0.7, 1.0)
                
            # Restart mechanism if stagnation detected
            if np.ptp(self.fitness) < 1e-6:
                self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
                for i in range(self.pop_size):
                    self.fitness[i] = self.evaluate(func, self.population[i])

        best_idx = np.argmin(self.fitness)
        return self.fitness[best_idx], self.population[best_idx]