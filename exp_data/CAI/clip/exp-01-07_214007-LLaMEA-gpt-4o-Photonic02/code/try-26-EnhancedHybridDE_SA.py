import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cross_prob = 0.9
        self.f_scale = 0.85
        self.temperature = 1.0
        self.cooling_rate = 0.90
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        ranked_indices = np.argsort(fitness)
        best_idx = ranked_indices[0]
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        eval_count = self.population_size

        while eval_count < self.budget:
            previous_best_fitness = best_fitness
            rank_weights = np.linspace(1, 0, self.population_size)
            for i in ranked_indices:
                if eval_count >= self.budget:
                    break

                indices = [idx for idx in ranked_indices if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                adaptive_scale = self.f_scale * (1 - eval_count / self.budget) * rank_weights[i]
                mutant = np.clip(a + adaptive_scale * (b - c), lb, ub)
                adaptive_cross_prob = self.cross_prob * (1 - eval_count / self.budget) * rank_weights[i]
                crossover_mask = np.random.rand(self.dim) < adaptive_cross_prob
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            improvement_rate = (previous_best_fitness - best_fitness) / abs(previous_best_fitness) if previous_best_fitness != 0 else 1
            self.temperature *= (self.cooling_rate + 0.05 * improvement_rate) * (1 - eval_count / self.budget)
            ranked_indices = np.argsort(fitness)

        return best_solution