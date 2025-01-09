import numpy as np

class HybridSADEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.temp_initial = 1000
        self.temp_final = 1
        self.cooling_rate = 0.95
        self.pop_size = 10

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + np.random.rand(self.pop_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.pop_size
        temperature = self.temp_initial
        
        while evaluations < self.budget and temperature > self.temp_final:
            # Simulated Annealing Step
            for i in range(self.pop_size):
                candidate = population[i] + np.random.normal(0, 1, self.dim) * temperature
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                evaluations += 1
                if candidate_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - candidate_fitness) / temperature):
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness

                if evaluations >= self.budget:
                    break

            temperature *= self.cooling_rate

            # Differential Evolution Step
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break
        
        return best_solution, best_fitness