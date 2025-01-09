import numpy as np

class HybridDESA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.init_population_size = 50  # Increased initial population
        self.F = 0.8  # Differential evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.temperature = 1.0  # Initial temperature for simulated annealing
        self.decay_rate = 0.97  # Temperature decay rate
        self.min_population_size = 10  # Minimum population size

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.init_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.init_population_size
        
        while eval_count < self.budget:
            current_population_size = max(self.min_population_size, 
                                          self.init_population_size - eval_count // (self.budget // 4))
            for i in range(current_population_size):
                idxs = [idx for idx in range(current_population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])

                trial = np.copy(population[i])
                cross_points = np.random.rand(self.dim) < self.CR
                trial[cross_points] = mutant[cross_points]

                trial_fitness = func(trial)
                eval_count += 1

                # Adaptive Simulated Annealing Acceptance
                acceptance_prob = np.exp((fitness[i] - trial_fitness) / (self.temperature / np.log(eval_count + 1)))
                if trial_fitness < fitness[i] or acceptance_prob > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

                if eval_count >= self.budget:
                    break

            self.temperature *= self.decay_rate

        return self.f_opt, self.x_opt