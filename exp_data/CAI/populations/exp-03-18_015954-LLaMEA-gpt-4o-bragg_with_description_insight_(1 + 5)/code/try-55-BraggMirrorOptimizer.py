import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def differential_evolution(self, bounds, pop_size=20, F=0.8, CR=0.9):
        population = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        best_solution = None
        best_value = float('inf')
        adaptive_F = F

        def levy_flight(Lambda=1.5):  # Changed to include Lévy flight
            sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
                     (np.math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
            u = np.random.normal(0, sigma, size=self.dim)
            v = np.random.normal(0, 1, size=self.dim)
            step = u / np.abs(v)**(1 / Lambda)
            return step

        def mutate(target_idx):
            idxs = [idx for idx in range(pop_size) if idx != target_idx]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            levy_steps = levy_flight()  # Changed to use Lévy flight mutation
            return np.clip(a + adaptive_F * (b - c) + levy_steps, bounds.lb, bounds.ub)

        def crossover(target, mutant):
            chaos = np.random.rand(self.dim)  # Changed to implement chaotic sequence
            cross_points = chaos < CR
            periodic_points = np.arange(0, self.dim, 2)
            cross_points[periodic_points] = True
            trial = np.where(cross_points, mutant, target)
            return trial

        def oppositional_learning():
            opp_population = bounds.lb + bounds.ub - population
            for i in range(pop_size):
                opp_value = self.func(opp_population[i])
                if opp_value < self.func(population[i]):
                    population[i] = opp_population[i]

        for gen in range(self.budget // pop_size):
            adaptive_F = 0.5 + (0.5 * np.sin(np.pi * gen / self.budget))
            CR = 0.7 + 0.2 * np.sin(2 * np.pi * gen / self.budget)
            pop_size = 20 + int(10 * np.sin(2 * np.pi * gen / self.budget))
            oppositional_learning()
            for i in range(pop_size):
                mutant = mutate(i)
                trial = crossover(population[i], mutant)
                trial_value = self.func(trial)

                if trial_value < best_value:
                    best_value = trial_value
                    best_solution = trial

                if trial_value < self.func(population[i]):
                    population[i] = trial

        return best_solution, best_value

    def local_search(self, solution):
        res = minimize(self.func, solution, bounds=[(lb, ub) for lb, ub in zip(self.bounds.lb, self.bounds.ub)], method='L-BFGS-B')
        return res.x

    def __call__(self, func):
        self.func = func
        self.bounds = func.bounds

        best_solution, best_value = self.differential_evolution(self.bounds)
        fine_tuned_solution = self.local_search(best_solution)

        if self.func(fine_tuned_solution) < best_value:
            best_solution = fine_tuned_solution

        return best_solution