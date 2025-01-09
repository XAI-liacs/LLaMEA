import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temp_initial = 100.0
        self.temp_final = 1.0
        self.cooling_rate = 0.95

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        eval_count = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        temperature = self.temp_initial

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                
                # Differential Evolution
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                
                trial = np.where(crossover, mutant, population[i])
                
                # Simulated Annealing acceptance
                trial_fitness = func(trial)
                eval_count += 1
                
                delta_fitness = trial_fitness - fitness[i]
                if delta_fitness < 0 or np.exp(-delta_fitness / temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                    if trial_fitness < best_fitness:
                        best_individual = trial.copy()
                        best_fitness = trial_fitness
            
            # Update temperature
            temperature = max(self.temp_final, temperature * self.cooling_rate)

        return best_individual, best_fitness