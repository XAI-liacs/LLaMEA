import numpy as np

class HybridDESA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.initial_temp = 1.0  # Initial temperature for Simulated Annealing
        self.final_temp = 0.01  # Final temperature for Simulated Annealing
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for eval_count in range(self.budget):
            temperature = self.initial_temp - (self.initial_temp - self.final_temp) * (eval_count / self.budget)

            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[i])

                # Simulated Annealing acceptance probability
                trial_fitness = func(trial)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = pop[i]
            
            if eval_count >= self.budget - self.pop_size:
                break

        return self.f_opt, self.x_opt