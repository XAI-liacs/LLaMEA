import numpy as np

class QuantumEnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_perturbation(self, target, best, trial, eval_count):
        lambda_factor = eval_count / self.budget
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        delta = np.random.rand(self.dim)
        new_trial = trial + quantum_factor * (best - target) * delta
        return new_trial

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + self.f * (pop[b] - pop[c])
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                trial = self.quantum_perturbation(pop[i], best, trial, eval_count)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial

                if eval_count >= self.budget:
                    break

        return best