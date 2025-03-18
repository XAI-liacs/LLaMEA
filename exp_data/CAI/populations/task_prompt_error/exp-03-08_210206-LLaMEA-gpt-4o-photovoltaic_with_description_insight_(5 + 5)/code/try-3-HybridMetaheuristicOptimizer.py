import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Population size for DE
        self.f = 0.8  # DE scale factor
        self.cr = 0.9  # Crossover probability
        self.local_search_prob = 0.3  # Probability to perform local search
        self.increment_step = 2  # Layer-wise complexity increment
        self.current_budget = 0

    def local_search(self, candidate, func, bounds):
        """Perform a simple local search around a candidate solution."""
        best_candidate = candidate.copy()
        best_score = func(candidate)
        for d in range(self.dim):
            perturbation = np.random.uniform(-0.1, 0.1) * (bounds.ub[d] - bounds.lb[d])
            new_candidate = candidate.copy()
            new_candidate[d] += perturbation
            new_candidate[d] = np.clip(new_candidate[d], bounds.lb[d], bounds.ub[d])
            score = func(new_candidate)
            if score > best_score:
                best_score = score
                best_candidate = new_candidate
            self.current_budget += 1
            if self.current_budget >= self.budget:
                break
        return best_candidate, best_score

    def differential_evolution(self, population, func, bounds):
        """Perform a single iteration of Differential Evolution."""
        new_population = []
        for i in range(len(population)):
            if self.current_budget >= self.budget:
                break
            a, b, c = np.random.choice(len(population), 3, replace=False)
            mutant = population[a] + self.f * (population[b] - population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            trial = np.copy(population[i])
            jrand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.cr or j == jrand:
                    trial[j] = mutant[j]
            trial_score = func(trial)
            self.current_budget += 1
            if trial_score > func(population[i]):
                new_population.append(trial)
            else:
                new_population.append(population[i])
        return np.array(new_population)

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        self.current_budget = self.population_size

        while self.current_budget < self.budget:
            population = self.differential_evolution(population, func, bounds)
            if np.random.rand() < self.local_search_prob:
                for i in range(len(population)):
                    if self.current_budget >= self.budget:
                        break
                    population[i], scores[i] = self.local_search(population[i], func, bounds)
            
            # If allowed, increment layer complexity gradually
            if self.dim + self.increment_step <= self.budget - self.current_budget:
                self.dim += self.increment_step

        best_idx = np.argmax(scores)
        return population[best_idx], scores[best_idx]