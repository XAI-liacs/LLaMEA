import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.adaptation_rate = 0.1

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        initial_eval = evaluations  # New: track initial evaluations

        while evaluations < self.budget:
            inertia_factor = 1 - (evaluations - initial_eval) / (self.budget - initial_eval)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                noise_scale = inertia_factor * 0.01 * (1 + np.var(fitness) / np.mean(fitness))
                dimension_scale = np.random.rand(self.dim)  # New: dimension-wise adaptive scaling
                stochastic_F = self.F * (1 + 0.1 * np.random.randn(self.dim))  # Changed: stochastic scaling in mutation
                mutant = np.clip(a + dimension_scale * stochastic_F * (b - c) + self.F * (population[i] - a) + np.random.randn(self.dim) * noise_scale, lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                if evaluations >= self.budget:
                    break
            
            self.F = 0.5 + (np.var(fitness) / np.max(fitness)) ** 0.5 * self.adaptation_rate
            self.F = np.clip(self.F * inertia_factor, 0.1, 1.0)

            successful_trials = fitness < np.roll(fitness, 1)
            self.CR = self.CR + self.adaptation_rate * (np.mean(successful_trials) - np.var(successful_trials) * np.mean(fitness) * 0.5)
            self.CR = np.clip(self.CR * (1 + np.var(fitness)), 0.1, 1.0)  # Modified: adapt CR based on fitness variance
            
            if evaluations % (self.budget // 10) == 0:
                self.population_size = max(5 * self.dim, self.population_size - int(0.1 * self.population_size))
                population = population[np.argsort(fitness)[:self.population_size]]
                fitness = fitness[np.argsort(fitness)[:self.population_size]]
                
            # Introduce adaptive mutation scaling based on best fitness
            best_fit = np.min(fitness)
            self.F = self.F * (1 + np.log(1 + best_fit))

        best_idx = np.argmin(fitness)
        return population[best_idx]