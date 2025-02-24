import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Adjusted based on problem size
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        # Initialize population with periodicity consideration
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        population[::2] = np.histogram(np.random.uniform(bounds[0], bounds[1], (self.population_size//2, self.dim)), bins=10)[0]  # Slightly changed line
        # Introduce periodicity
        population = self.introduce_periodicity(population)
        fitness = np.array([func(ind) for ind in population])
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            new_population = np.zeros_like(population)

            for i in range(self.population_size):
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                self.f = 0.8 * (1 - (eval_count / self.budget))  # Adapt mutation scaling
                mutant = np.clip(a + self.f * (b - c), bounds[0], bounds[1])
                
                # Crossover (dynamically adjusted)
                self.cr = 0.9 * (1 - (eval_count / self.budget))
                crossover = np.random.rand(self.dim) < self.cr
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, population[i])
                trial = self.introduce_periodicity(trial.reshape(1, -1))[0]  # Maintain periodicity
                
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

    def introduce_periodicity(self, population):
        # Encourage periodicity by averaging segment thicknesses
        period = self.dim // 2
        for i in range(len(population)):
            for j in range(0, self.dim, period):
                segment = population[i, j:j + period]
                population[i, j:j + period] = np.mean(segment) * (1 + 0.1 * np.sin(j))  # Adaptive periodicity
        return population