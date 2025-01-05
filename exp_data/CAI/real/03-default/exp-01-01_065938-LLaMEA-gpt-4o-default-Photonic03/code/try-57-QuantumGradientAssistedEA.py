import numpy as np

class QuantumGradientAssistedEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.9
        self.gradient_weight = 0.1
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_quantum = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = lb + (ub - lb) * np.sin(np.pi * population_quantum)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        def approximate_gradient(x):
            epsilon = 1e-5
            grad = np.zeros_like(x)
            for j in range(self.dim):
                x_eps = np.copy(x)
                x_eps[j] += epsilon
                grad[j] = (func(x_eps) - func(x)) / epsilon
            return grad

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]

                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])

                gradient = approximate_gradient(trial)
                trial = trial - self.gradient_weight * gradient
                
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    next_pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = trial
                else:
                    next_pop[i] = pop[i]

            pop = next_pop
            self.history.append(best_global)

        return best_global