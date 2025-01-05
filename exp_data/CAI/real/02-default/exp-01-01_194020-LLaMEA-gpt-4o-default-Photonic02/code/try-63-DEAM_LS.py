import numpy as np

class DEAM_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.best_solution = None
        self.best_value = float('inf')
    
    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
    
    def mutate(self, population):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[idxs]
        return a + self.mutation_factor * (b - c)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
    
    def local_search(self, candidate, lb, ub):
        perturbation = (ub - lb) * (np.random.rand(self.dim) - 0.5) * 0.1
        new_candidate = candidate + perturbation
        return np.clip(new_candidate, lb, ub)
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            new_population = []
            for target_idx in range(self.population_size):
                target = population[target_idx]
                mutant = self.mutate(population)
                mutant = np.clip(mutant, lb, ub)
                trial = self.crossover(target, mutant)
                trial_value = func(trial)
                evaluations += 1
                
                if trial_value < func(target):
                    new_population.append(trial)
                    if trial_value < self.best_value:
                        self.best_value = trial_value
                        self.best_solution = trial
                else:
                    new_population.append(target)
                    
                if evaluations >= self.budget:
                    break
                
                if np.random.rand() < 0.1:  # Local search probability
                    local_candidate = self.local_search(new_population[-1], lb, ub)
                    local_value = func(local_candidate)
                    evaluations += 1
                    if local_value < func(new_population[-1]):
                        new_population[-1] = local_candidate
                        if local_value < self.best_value:
                            self.best_value = local_value
                            self.best_solution = local_candidate
                
            population = np.array(new_population)
        
        return self.best_solution, self.best_value