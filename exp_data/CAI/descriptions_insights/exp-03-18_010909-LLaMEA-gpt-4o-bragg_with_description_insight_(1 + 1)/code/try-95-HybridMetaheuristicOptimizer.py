import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.85
        self.CR = 0.9
        self.local_search_probability = 0.5
        self.penalty_factor = 100.0
        self.dynamic_pop_size = lambda evals: max(5, int(self.pop_size * (1 - evals / self.budget)))  # Dynamic pop size

    def differential_evolution(self, func, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.pop_size

        while num_evals < self.budget:
            new_population = np.copy(population)
            current_pop_size = self.dynamic_pop_size(num_evals)
            for i in range(current_pop_size):
                a, b, c = population[np.random.choice(current_pop_size, 3, replace=False)]
                adaptive_F = self.F * (np.sin(np.pi * num_evals / self.budget))**1.2
                mutant = np.clip(a + adaptive_F * (b - c), bounds.lb, bounds.ub)
                dynamic_CR = self.CR * (1 - num_evals / self.budget) + 0.1 * (num_evals / self.budget)
                cross_points = np.random.rand(self.dim) < dynamic_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial = self.frequency_domain_transform(trial)  # Apply frequency domain transformation

                periodicity_penalty = self.calculate_periodicity_penalty(trial) / self.dim
                trial_fitness = func(trial) + 0.3 * periodicity_penalty * (1 - fitness[i] / np.min(fitness))  # Adjusted penalty weight
                num_evals += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                adaptive_local_search_prob = self.local_search_probability * np.exp(-4.0 * num_evals/self.budget)  # Adjusted decay factor
                if np.random.rand() < adaptive_local_search_prob:
                    local_res = minimize(func, trial, method='L-BFGS-B', bounds=np.array(list(zip(bounds.lb, bounds.ub))))
                    local_fitness = local_res.fun
                    num_evals += local_res.nfev
                    if local_fitness < fitness[i]:
                        new_population[i] = local_res.x
                        fitness[i] = local_fitness

                if num_evals >= self.budget:
                    break
            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def frequency_domain_transform(self, solution):
        transformed = np.fft.ifft(np.fft.fft(solution) * np.exp(-0.1 * np.arange(self.dim)))
        return np.real(transformed)

    def calculate_periodicity_penalty(self, solution):
        period = 2
        penalty = 0
        for i in range(0, len(solution) - period):
            penalty += np.sum((solution[i:i+period] - solution[i+period:i+2*period])**2)**1.5
        return self.penalty_factor * penalty

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_fitness = self.differential_evolution(func, bounds)
        return best_solution, best_fitness