import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 + int(2 * np.sqrt(dim))
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # differential weight
        self.CR = 0.9  # crossover probability
    
    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        
        while evals < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if evals >= self.budget:
                    break
            
            # Local search phase
            best_idx = np.argmin(fitness)
            best_ind = population[best_idx]
            step_size = (self.upper_bound - self.lower_bound) * 0.1
            for _ in range(5):  # Attempt 5 local refinements
                perturbation = np.random.uniform(-step_size, step_size, self.dim)
                candidate = np.clip(best_ind + perturbation, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                evals += 1
                
                if candidate_fitness < fitness[best_idx]:
                    population[best_idx] = candidate
                    fitness[best_idx] = candidate_fitness
                
                if evals >= self.budget:
                    break
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]