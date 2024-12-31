import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, max_restart=5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.max_restart = max_restart
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        eval_count = 0
        restart_count = 0

        while eval_count < self.budget and restart_count < self.max_restart:
            # Initialize population
            population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            eval_count += self.population_size

            best_idx = np.argmin(fitness)
            best_fitness = fitness[best_idx]
            best_individual = population[best_idx]

            if best_fitness < self.f_opt:
                self.f_opt = best_fitness
                self.x_opt = best_individual

            # Differential Evolution parameters
            F_base = 0.8  # Differential weight
            CR = 0.9  # Crossover probability

            for _ in range(self.budget // self.population_size):
                if eval_count >= self.budget:
                    break
                new_population = np.zeros_like(population)
                new_fitness = np.zeros(self.population_size)

                for i in range(self.population_size):
                    # Adaptive mutation factor
                    F = F_base + 0.2 * np.random.rand() - 0.1

                    # Mutation
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                    # Crossover
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])

                    # Selection
                    trial_fitness = func(trial)
                    eval_count += 1

                    if trial_fitness < fitness[i]:
                        new_population[i] = trial
                        new_fitness[i] = trial_fitness
                    else:
                        new_population[i] = population[i]
                        new_fitness[i] = fitness[i]

                if np.min(new_fitness) < self.f_opt:
                    self.f_opt = np.min(new_fitness)
                    self.x_opt = new_population[np.argmin(new_fitness)]
                
                # Update population and fitness
                population = new_population
                fitness = new_fitness

                # Dynamic population resizing
                if eval_count > self.budget * 0.5 and eval_count < self.budget * 0.75:
                    self.population_size = int(self.population_size * 0.9)

                # Restart mechanism
                if eval_count >= self.budget * (restart_count + 1) / self.max_restart:
                    restart_count += 1
                    break

        return self.f_opt, self.x_opt