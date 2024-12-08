import numpy as np

class QuantumAdaptiveNeighborhoodOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 12 * dim  # Adjusted population size
        self.max_iter = budget // self.pop_size
        self.F_base = 0.5  # Adjusted differential weight
        self.CR_base = 0.9  # Adjusted crossover probability
        self.w_base = 0.5  # Adjusted inertia weight for PSO
        self.c1 = 1.5  # Adjusted personal attraction coefficient
        self.c2 = 1.5  # Adjusted global attraction coefficient
        self.adaptivity = 0.1  # Learning rate for adaptivity

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        fitness = np.apply_along_axis(func, 1, pop)
        personal_best = pop.copy()
        personal_best_fitness = fitness.copy()
        global_best = pop[np.argmin(fitness)]
        
        eval_count = self.pop_size
        
        for t in range(self.max_iter):
            learning_rate = self.adaptivity * np.exp(-t / self.max_iter)
            neighbor_influence = np.mean(pop, axis=0)
            adaptive_F = self.F_base + learning_rate

            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 4, replace=False)
                x0, x1, x2, x3 = pop[indices]
                mutant = x0 + adaptive_F * (x1 - x2) + adaptive_F * (x3 - x0)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR_base
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < func(global_best):
                            global_best = trial

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocity = (self.w_base * velocity + 
                        self.c1 * r1 * (personal_best - pop) + 
                        self.c2 * r2 * (global_best - pop) + 
                        neighbor_influence * learning_rate)
            pop = np.clip(pop + velocity, self.lower_bound, self.upper_bound)
            
            fitness = np.apply_along_axis(func, 1, pop)
            eval_count += self.pop_size
            
            better_mask = fitness < personal_best_fitness
            personal_best[better_mask] = pop[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]
            if np.min(fitness) < func(global_best):
                global_best = pop[np.argmin(fitness)]

            if eval_count >= self.budget:
                break

        return global_best