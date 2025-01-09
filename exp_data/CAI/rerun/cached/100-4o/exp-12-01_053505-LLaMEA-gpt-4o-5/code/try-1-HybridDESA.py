import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = max(5, dim * 5)  # Population size
        self.temperature = 1000  # Initial temperature for annealing
        self.cooling_rate = 0.99  # Cooling rate for annealing
    
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            # Differential Evolution mutation
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                # Adaptive mutation factor
                F = 0.5 + np.random.rand() * 0.5
                mutant = np.clip(x0 + F * (x1 - x2), self.lb, self.ub)
                # Crossover
                crossover = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover, mutant, population[i])
                
                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Simulated Annealing
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                candidate = population[i] + np.random.normal(0, 0.1, self.dim)
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = func(candidate)
                evaluations += 1

                # Metropolis acceptance criterion
                if candidate_fitness < fitness[i] or \
                   np.random.rand() < np.exp((fitness[i] - candidate_fitness) / self.temperature):
                    population[i] = candidate
                    fitness[i] = candidate_fitness

            # Cool down
            self.temperature *= self.cooling_rate
        
        # Return the best found solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

# Example usage:
# optimizer = HybridDESA(budget=1000, dim=10)
# best_solution, best_value = optimizer(my_black_box_function)