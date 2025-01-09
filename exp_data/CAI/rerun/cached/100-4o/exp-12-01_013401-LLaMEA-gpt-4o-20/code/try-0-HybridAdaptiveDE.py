import numpy as np
from scipy.optimize import minimize

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Define based on problem complexity
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.eval_count = 0

    def differential_evolution_step(self, population, scores):
        new_population = np.copy(population)
        for i in range(self.population_size):
            # Select three random individuals
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            
            # Mutation
            mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            
            # Crossover
            crossover = np.random.rand(self.dim) < self.crossover_probability
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            trial = np.where(crossover, mutant, population[i])
            
            # Selection
            trial_score = func(trial)
            self.eval_count += 1
            if trial_score < scores[i]:
                new_population[i] = trial
                scores[i] = trial_score

            if self.eval_count >= self.budget:
                break
        return new_population, scores
    
    def nelder_mead_local_search(self, best_individual):
        res = minimize(func, best_individual, method='Nelder-Mead', bounds=[(self.lower_bound, self.upper_bound)] * self.dim, 
                       options={'maxfev': self.budget - self.eval_count, 'xatol': 1e-6, 'fatol': 1e-6})
        self.eval_count += res.nfev
        return res.x, res.fun

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            population, scores = self.differential_evolution_step(population, scores)
            best_idx = np.argmin(scores)
            best_individual, best_score = population[best_idx], scores[best_idx]
            
            if self.eval_count < self.budget:
                # Local search using Nelder-Mead
                best_individual, best_score = self.nelder_mead_local_search(best_individual)

            # Update best known positions and scores
            population[best_idx] = best_individual
            scores[best_idx] = best_score

            if self.eval_count >= self.budget:
                break

        return population[np.argmin(scores)], np.min(scores)