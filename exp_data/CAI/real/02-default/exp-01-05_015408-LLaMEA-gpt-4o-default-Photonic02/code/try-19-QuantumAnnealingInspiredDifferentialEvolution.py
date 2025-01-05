import numpy as np

class QuantumAnnealingInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cooling_rate = 0.9
        self.initial_temperature = 100.0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        temperature = self.initial_temperature
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index].copy()

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = a + self.mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i] or np.exp((fitness[i] - trial_fitness) / temperature) > np.random.rand():
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < fitness[best_index]:
                    best_index = i
                    best_solution = trial_vector.copy()

                if evaluations >= self.budget:
                    break
            
            temperature *= self.cooling_rate

        return best_solution, fitness[best_index]