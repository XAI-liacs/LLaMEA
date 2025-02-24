import numpy as np
from scipy.optimize import minimize

class ADESOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 * dim  # Smaller, more focused population
        self.f = 0.5  # Initial differential weight
        self.cr = 0.7  # Initial crossover probability

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        # Initialize population with symmetry considerations
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        population = self.introduce_symmetry(population)
        fitness = np.array([func(ind) for ind in population])
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            new_population = np.zeros_like(population)

            sorted_indices = np.argsort(fitness)
            for i in range(self.population_size):
                # Mutation with swarm influence
                g_best = population[sorted_indices[0]]  # Global best
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                self.f = 0.5 + 0.3 * (fitness[sorted_indices[-1]] - fitness[i]) / (fitness[sorted_indices[-1]] - fitness[sorted_indices[0]] + 1e-9)
                mutant = np.clip(g_best + self.f * (b - c), bounds[0], bounds[1])
                
                # Crossover with adaptive adjustments
                self.cr = 0.7 * (1 - fitness[i] / (fitness[sorted_indices[0]] + 1e-9))
                crossover = np.random.rand(self.dim) < self.cr
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, population[i])
                trial = self.introduce_symmetry(trial.reshape(1, -1))[0]  # Maintain symmetry
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness > fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

                if eval_count >= self.budget:
                    break

            population = new_population
        
        # Local optimization on the best found solution
        best_index = np.argmax(fitness)
        best_solution = population[best_index]
        
        # Use a local optimizer from scipy for fine-tuning
        result = minimize(func, best_solution, bounds=bounds.T, method='L-BFGS-B', options={'disp': True})
        
        return result.x

    def introduce_symmetry(self, population):
        # Encourage symmetry by averaging half-segment thicknesses
        half_dim = self.dim // 2
        for i in range(len(population)):
            for j in range(half_dim):
                avg_value = (population[i, j] + population[i, -j-1]) / 2
                population[i, j] = avg_value * (1 + 0.1 * np.cos(j))
                population[i, -j-1] = avg_value * (1 + 0.1 * np.cos(j))
        return population