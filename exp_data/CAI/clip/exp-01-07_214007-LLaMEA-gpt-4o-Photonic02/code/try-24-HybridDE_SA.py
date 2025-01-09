import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cross_prob = 0.9  # Changed from 0.7 to 0.9
        self.f_scale = 0.85  # Changed from 0.8 to 0.85
        self.temperature = 1.0
        self.cooling_rate = 0.88  # Adjusted cooling rate from 0.90 to 0.88
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        eval_count = self.population_size
        stable_count = 0  # Track stability count

        while eval_count < self.budget:
            previous_best_fitness = best_fitness  # Track previous best fitness for improvement rate
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                adaptive_scale = self.f_scale * (1 - eval_count / self.budget)
                mutant = np.clip(a + adaptive_scale * (b - c), lb, ub)
                self.cross_prob = 0.85 + 0.15 * (1 - eval_count / self.budget)  # Adaptive crossover probability
                
                crossover_mask = np.random.rand(self.dim) < self.cross_prob
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                        stable_count = 0  # Reset stable count on improvement
                    else:
                        stable_count += 1  # Increment stable count if no improvement

            improvement_rate = (previous_best_fitness - best_fitness) / abs(previous_best_fitness) if previous_best_fitness != 0 else 1
            self.temperature *= (self.cooling_rate + 0.05 * improvement_rate) * (1 - eval_count / self.budget)
            
            # Adjust f_scale based on stable count to encourage exploration
            if stable_count > 5:
                self.f_scale = min(self.f_scale * 1.1, 0.9)
        
        return best_solution