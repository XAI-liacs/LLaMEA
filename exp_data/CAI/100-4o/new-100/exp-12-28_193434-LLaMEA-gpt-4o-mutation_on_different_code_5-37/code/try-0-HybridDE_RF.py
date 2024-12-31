import numpy as np
from sklearn.ensemble import RandomForestRegressor

class HybridDE_RF:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.pop_size = 15 * dim
        self.f_opt = np.inf
        self.x_opt = None
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.surrogate = RandomForestRegressor()

    def differential_evolution_step(self, func, F=0.5, CR=0.9):
        for i in range(self.pop_size):
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            if trial_fitness < func(self.population[i]):
                self.population[i] = trial
                if trial_fitness < self.f_opt:
                    self.f_opt, self.x_opt = trial_fitness, trial

    def build_surrogate_model(self, func, sample_size=100):
        X = np.random.uniform(self.bounds[0], self.bounds[1], (sample_size, self.dim))
        y = np.array([func(x) for x in X])
        self.surrogate.fit(X, y)

    def exploit_with_surrogate(self, func, num_tries=50):
        for _ in range(num_tries):
            x = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            predicted_fitness = self.surrogate.predict(x.reshape(1, -1))
            if predicted_fitness < self.f_opt:
                actual_fitness = func(x)
                if actual_fitness < self.f_opt:
                    self.f_opt, self.x_opt = actual_fitness, x

    def __call__(self, func):
        evaluations = 0

        # Initial evaluations of the population
        fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size

        # Find the best individual in the initial population
        best_idx = np.argmin(fitness)
        self.f_opt, self.x_opt = fitness[best_idx], self.population[best_idx]

        # Main optimization loop
        while evaluations < self.budget:
            self.differential_evolution_step(func)
            evaluations += self.pop_size

            if evaluations + 100 <= self.budget:  # Check if we have room for surrogate exploitation
                self.build_surrogate_model(func)
                evaluations += 100
                self.exploit_with_surrogate(func)
        
        return self.f_opt, self.x_opt