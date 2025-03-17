import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.temperature = 1.0
        self.cooling_rate = 0.99
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Differential Evolution Mutation
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutation_factor = 0.5 + 0.3 * np.sin(np.random.rand() * np.pi)
                mutant = np.clip(population[a] + mutation_factor * (population[b] - population[c]), lb, ub)
                mutant += self._levy_flight()  # Levy Flight integration
                
                # Simulated Annealing Crossover
                trial = np.where(np.random.rand(self.dim) < self.temperature, mutant, population[i])
                
                # Evaluate new candidate
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            # Dynamic Cooling Schedule
            self.temperature *= self.cooling_rate * (1 - evaluations / self.budget)
        
        return best_solution

    def _levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v)**(1 / beta)
        return step