import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 15 + int(self.dim * np.log(self.dim)))  # Increased population size for more diversity
        self.mutation_factor = 0.5 + np.random.rand(self.population_size) * 0.4  # Further widened mutation factor range
        self.crossover_rate = 0.5 + np.random.rand(self.population_size) * 0.4  # Adjusted crossover rate for balance
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0
        self.memory = self.population.copy()
        self.quantum_operator = 0.1  # Quantum influence parameter

    def __call__(self, func):
        self.evaluate_population(func)
        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index].copy()
        best_fitness = self.fitness[best_index]

        while self.eval_count < self.budget:
            current_population_size = min(self.population_size, self.budget - self.eval_count)
            
            for i in range(current_population_size):
                if self.eval_count >= self.budget:
                    break

                indices = np.random.choice(current_population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                if np.random.rand() < 0.07:  # Increased probability for adaptive memory utilization
                    memory_idx = np.random.choice(self.population_size)
                    quantum_effect = self.quantum_operator * (np.random.rand(self.dim) - 0.5) * (self.memory[memory_idx] - best_solution)
                else:
                    quantum_effect = np.zeros(self.dim)
                mutant_vector = x0 + self.mutation_factor[i] * (x1 - x2) + quantum_effect
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate[i], mutant_vector, self.population[i])

                trial_fitness = func(trial_vector)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial_vector.copy()
                        self.memory[i] = trial_vector.copy()

        return best_solution

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.eval_count += 1