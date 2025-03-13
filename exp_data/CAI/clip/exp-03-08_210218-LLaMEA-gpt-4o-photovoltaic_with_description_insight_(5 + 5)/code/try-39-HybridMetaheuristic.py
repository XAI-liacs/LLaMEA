import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, population_size, max_iter, F=0.8, CR=0.9, num_swarms=2):  # Add multi-swarm
        swarms = [np.random.rand(population_size, self.dim) for _ in range(num_swarms)]  # Initialize multiple swarms
        pop_fitness = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        self.evaluations += population_size * num_swarms

        for iteration in range(max_iter):
            CR = 0.5 + 0.5 * (iteration / max_iter)
            dynamic_F = 0.5 + 0.3 * (np.sin(iteration / max_iter * np.pi))
            for swarm_idx, pop in enumerate(swarms):
                for i in range(population_size):
                    indices = np.random.choice(population_size, 3, replace=False)
                    x0, x1, x2 = pop[indices]
                    mutant = np.clip(x0 + dynamic_F * (x1 - x2), bounds.lb, bounds.ub)
                    crossover = np.random.rand(self.dim) < CR
                    trial = np.where(crossover, mutant, pop[i])
                    
                    if self.evaluations < self.budget:
                        trial_fitness = func(trial)
                        self.evaluations += 1
                        if trial_fitness < pop_fitness[swarm_idx][i]:
                            pop[i] = trial
                            pop_fitness[swarm_idx][i] = trial_fitness

                    if self.evaluations >= self.budget:
                        break

                # Share best solution among swarms
                best_indices = [np.argmin(fitness) for fitness in pop_fitness]
                best_solutions = [swarm[best_idx] for swarm, best_idx in zip(swarms, best_indices)]
                for swarm in swarms:
                    for individual in swarm:
                        if np.random.rand() < 0.1:  # 10% probability to replace with best from other swarm
                            individual[:] = best_solutions[np.random.randint(num_swarms)]

            if self.evaluations >= self.budget:
                break

        best_solutions = [swarm[np.argmin(fitness)] for swarm, fitness in zip(swarms, pop_fitness)]
        return min(best_solutions, key=func)

    def local_refinement(self, func, solution, bounds):
        result = minimize(func, solution, method='Nelder-Mead', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)],
                          options={'maxfev': self.budget - self.evaluations, 'adaptive': True})
        self.evaluations += result.nfev
        return result.x if result.success else solution

    def __call__(self, func):
        bounds = func.bounds
        population_size = 10 + self.dim
        max_iter = 100
        phase_budget = self.budget // 2
        
        best_solution = self.differential_evolution(func, bounds, population_size, max_iter)
        if self.evaluations < self.budget:
            best_solution = self.local_refinement(func, best_solution, bounds)
        
        return best_solution