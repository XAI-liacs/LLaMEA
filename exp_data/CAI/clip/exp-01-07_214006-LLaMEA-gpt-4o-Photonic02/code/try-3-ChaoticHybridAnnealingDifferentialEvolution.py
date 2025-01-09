import numpy as np

class ChaoticHybridAnnealingDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.current_evaluations = 0
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.chaos_map = self.logistic_map
        self.chaos_param = 0.7

    def logistic_map(self, x, r=3.99):
        return r * x * (1 - x)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        
        while self.current_evaluations < self.budget:
            # Simulated annealing step with chaotic cooling
            for i in range(self.population_size):
                candidate = population[i] + self.temperature * (np.random.rand(self.dim) - 0.5)
                candidate = np.clip(candidate, bounds[:,0], bounds[:,1])
                candidate_fitness = func(candidate)
                self.current_evaluations += 1
                
                if candidate_fitness < fitness[i] or np.exp((fitness[i] - candidate_fitness) / self.temperature) > np.random.rand():
                    population[i] = candidate
                    fitness[i] = candidate_fitness
            
            self.temperature *= self.cooling_rate * self.chaos_map(self.chaos_param)
            self.chaos_param = self.chaos_map(self.chaos_param)
            
            # Differential evolution step with chaotic crossover
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + 0.8 * (x1 - x2), bounds[:,0], bounds[:,1])
                crossover_prob = 0.75 + 0.2 * self.chaos_map(self.chaos_param)
                cross_points = np.random.rand(self.dim) < crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.current_evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            if self.current_evaluations >= self.budget:
                break

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]