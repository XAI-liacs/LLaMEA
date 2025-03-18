import numpy as np
from scipy.optimize import minimize

class HybridSQODEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.bounds = None

    def _initialize_population(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = lb + ub - pop  # Quasi-oppositional initialization
        combined_pop = np.vstack((pop, opp_pop))
        return combined_pop

    def _evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def _differential_evolution_step(self, pop, fitness, best_individual):
        new_pop = []
        for i in range(len(pop)):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            mutant = np.clip(a + 0.8 * (b - c), self.bounds.lb, self.bounds.ub)
            cross_points = np.random.rand(self.dim) < 0.9
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial_fitness = self._evaluate_population(func, [trial])[0]
            if trial_fitness < fitness[i]:
                new_pop.append(trial)
            else:
                new_pop.append(pop[i])
        return np.array(new_pop)

    def _local_optimization(self, func, best_individual):
        result = minimize(func, best_individual, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return result.x, result.fun

    def __call__(self, func):
        self.bounds = func.bounds
        pop = self._initialize_population(self.bounds.lb, self.bounds.ub)
        fitness = self._evaluate_population(func, pop)
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = 2 * self.population_size
        while evaluations < self.budget:
            pop = self._differential_evolution_step(pop, fitness, best_individual)
            fitness = self._evaluate_population(func, pop)
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_individual = pop[current_best_idx]
                best_fitness = fitness[current_best_idx]

            # Local optimization every few generations
            if evaluations % (self.population_size * 3) == 0:
                local_best, local_best_fitness = self._local_optimization(func, best_individual)
                if local_best_fitness < best_fitness:
                    best_individual, best_fitness = local_best, local_best_fitness
            
            evaluations += self.population_size

        return best_individual