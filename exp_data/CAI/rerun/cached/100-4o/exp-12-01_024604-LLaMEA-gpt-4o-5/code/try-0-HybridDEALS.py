import numpy as np

class HybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f_lower_bound = -5.0
        self.f_upper_bound = 5.0
        self.population_size = max(5, 10 * dim)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.local_search_prob = 0.2  # Probability of performing a local search

    def __call__(self, func):
        pop = self.f_lower_bound + (self.f_upper_bound - self.f_lower_bound) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in pop])
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                candidates = [c for c in range(self.population_size) if c != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), self.f_lower_bound, self.f_upper_bound)
                
                trial = np.array([mutant[j] if np.random.rand() < self.CR else pop[i][j] for j in range(self.dim)])
                
                if np.random.rand() < self.local_search_prob:
                    trial = self.adaptive_local_search(trial, func)
                
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                
                if evals >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return pop[best_index]

    def adaptive_local_search(self, solution, func):
        step_size = (self.f_upper_bound - self.f_lower_bound) * 0.05
        best_local = solution
        best_fitness = func(solution)

        for _ in range(5):
            candidate = best_local + step_size * (np.random.rand(self.dim) - 0.5)
            candidate = np.clip(candidate, self.f_lower_bound, self.f_upper_bound)
            candidate_fitness = func(candidate)

            if candidate_fitness < best_fitness:
                best_fitness = candidate_fitness
                best_local = candidate

        return best_local