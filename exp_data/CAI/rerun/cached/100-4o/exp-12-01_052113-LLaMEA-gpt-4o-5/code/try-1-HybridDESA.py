import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.temperature = 1.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))

    def __call__(self, func):
        eval_count = 0
        fitness = np.array([func(ind) for ind in self.population])
        eval_count += self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                # Differential Evolution Mutation
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Adjust crossover probability based on progress
                progress_ratio = eval_count / self.budget
                crossover_probability = self.crossover_probability * (1 - progress_ratio)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                eval_count += 1
                
                # Selection and Simulated Annealing Acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    self.population[i] = trial
                    fitness[i] = trial_fitness
            
            # Cooling schedule for Simulated Annealing
            self.temperature *= 0.99
        
        best_index = np.argmin(fitness)
        return self.population[best_index], fitness[best_index]