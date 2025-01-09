import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 10 * dim)
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.1, 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0
        self.memory = np.copy(self.population)  # Initialize memory for learning

    def evaluate(self, func, individual):
        if self.eval_count < self.budget:
            self.eval_count += 1
            return func(individual)
        else:
            return np.inf

    def adapt_parameters(self, generation):
        learning_factor = np.exp(-generation / (0.5 * self.budget))  # Decay factor
        return (
            self.F_min + learning_factor * (self.F_max - self.F_min),
            self.CR_min + (self.CR_max - self.CR_min) * np.cos(generation * np.pi / (2 * self.budget))
        )

    def __call__(self, func):
        generation = 0
        while self.eval_count < self.budget:
            if generation == 0:
                for i in range(self.population_size):
                    self.fitness[i] = self.evaluate(func, self.population[i])
            new_population = np.copy(self.population)
            F, CR = self.adapt_parameters(generation)
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, self.population[i])
                trial_fitness = self.evaluate(func, trial_vector)
                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial_vector
                    self.fitness[i] = trial_fitness
                    self.memory[i] = trial_vector  # Update memory with successful trial
                else:
                    trial_vector = self.memory[i]  # Use memory to guide search if trial fails
                    trial_fitness = self.evaluate(func, trial_vector)
                    if trial_fitness < self.fitness[i]:
                        new_population[i] = trial_vector
                        self.fitness[i] = trial_fitness
            self.population = new_population
            generation += 1
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]