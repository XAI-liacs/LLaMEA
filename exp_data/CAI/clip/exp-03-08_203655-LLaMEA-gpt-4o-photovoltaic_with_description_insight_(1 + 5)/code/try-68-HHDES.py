import numpy as np

class HHDES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25  # Increased population size for diversity
        self.CR = 0.9
        self.F = 0.8
        self.pbest_fraction = 0.2
        self.population = np.random.rand(self.pop_size, dim)
        self.best_solution = np.random.rand(dim)  
        self.best_fitness = np.inf
        self.evaluate_budget = 0

    def evaluate(self, func, candidate):
        self.evaluate_budget += 1
        return func(candidate)

    def differential_evolution(self, func):
        for _ in range(self.budget):
            if self.evaluate_budget >= self.budget:
                break
            for i in range(self.pop_size):
                if self.evaluate_budget >= self.budget:
                    break
                donor = self.adaptive_mutate(i)
                trial = self.adaptive_crossover(self.population[i], donor)
                trial_fitness = self.evaluate(func, trial)
                if trial_fitness < self.evaluate(func, self.population[i]):
                    self.population[i] = trial
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
   
    def refine_population(self, func):
        new_pop = []
        for candidate in self.population:
            refined_candidate, _ = self.local_search(func, candidate)
            new_pop.append(refined_candidate)
        self.population = np.array(new_pop)

    def adaptive_mutate(self, current_idx):
        indices = list(range(self.pop_size))
        indices.remove(current_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = 0.4 + np.random.rand() * 0.5  
        mutant = self.population[a] + F * (self.population[b] - self.population[c])
        return np.clip(mutant, 0, 1)

    def adaptive_crossover(self, target, donor):
        CR = 0.6 + np.random.rand() * 0.4  
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, donor, target)

    def local_search(self, func, solution):
        perturbation = np.random.normal(0, 0.05, solution.shape) 
        candidate = solution + perturbation
        candidate = np.clip(candidate, 0, 1)
        candidate_fitness = self.evaluate(func, candidate)
        if candidate_fitness < self.evaluate(func, solution):
            return candidate, candidate_fitness
        return solution, self.evaluate(func, solution)

    def optimize_layers(self, func):
        for _ in range(self.budget // self.dim):
            if self.evaluate_budget >= self.budget:
                break
            self.differential_evolution(func)
            self.refine_population(func)  # Additional refinement step

    def __call__(self, func):
        self.optimize_layers(func)
        return self.best_solution