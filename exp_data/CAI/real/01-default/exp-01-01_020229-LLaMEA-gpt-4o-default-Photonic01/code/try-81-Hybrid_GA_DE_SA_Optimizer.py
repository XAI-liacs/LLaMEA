import numpy as np

class Hybrid_GA_DE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Suitable for genetic diversity
        self.crossover_rate = 0.7
        self.differential_weight = 0.8
        self.temperature = 1.0
        self.cooling_rate = 0.995
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_value = fitness[best_idx]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = a + self.differential_weight * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                
                trial_value = func(trial_vector)
                evaluations += 1
                
                if trial_value < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_value
                    
                    if trial_value < best_value:
                        best_individual = trial_vector
                        best_value = trial_value

                if evaluations >= self.budget:
                    break

            # Simulated Annealing component
            for i in range(self.population_size):
                perturbed_vector = population[i] + np.random.normal(0, self.temperature, self.dim)
                perturbed_vector = np.clip(perturbed_vector, lb, ub)
                perturbed_value = func(perturbed_vector)
                evaluations += 1

                if perturbed_value < fitness[i] or np.random.rand() < np.exp((fitness[i] - perturbed_value) / self.temperature):
                    population[i] = perturbed_vector
                    fitness[i] = perturbed_value

                    if perturbed_value < best_value:
                        best_individual = perturbed_vector
                        best_value = perturbed_value

                if evaluations >= self.budget:
                    break
        
            # Update temperature for Simulated Annealing
            self.temperature *= self.cooling_rate

        return best_individual, best_value