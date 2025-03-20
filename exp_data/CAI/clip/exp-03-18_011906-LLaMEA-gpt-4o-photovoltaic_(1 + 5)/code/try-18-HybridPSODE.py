import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * dim
        self.c1 = 1.5  # Adjusted cognitive coefficient
        self.c2 = 1.5  # Adjusted social coefficient
        self.w = 0.7
        self.mutation_factor = 0.8
        self.recombination_rate = 0.9
        self.particles = None
        self.velocities = None
        self.local_best = None
        self.global_best = None

    def chaotic_initialization(self, lb, ub):
        r = np.random.rand(self.population_size, self.dim)
        return lb + (ub - lb) * np.sin(np.pi * r)

    def adaptive_learning_rate(self, iteration, max_iter):
        return 0.5 + 0.5 * (1 - iteration / max_iter)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.particles = self.chaotic_initialization(lb, ub)
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in self.particles])
        self.local_best = self.particles.copy()
        self.global_best = self.particles[np.argmin(fitness)]
        local_best_fitness = fitness.copy()
        global_best_fitness = np.min(fitness)

        evaluations = self.population_size
        iteration = 0
        max_iter = self.budget // self.population_size

        while evaluations < self.budget:
            iteration += 1
            adaptive_c1 = self.adaptive_learning_rate(iteration, max_iter)
            adaptive_c2 = self.adaptive_learning_rate(iteration, max_iter)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.w = 0.9 - 0.5 * evaluations / self.budget
            self.velocities = (self.w * self.velocities +
                               adaptive_c1 * r1 * (self.local_best - self.particles) +
                               adaptive_c2 * r2 * (self.global_best - self.particles))
            self.particles = self.particles + self.velocities
            self.particles = np.clip(self.particles, lb, ub)

            new_fitness = np.array([func(x) for x in self.particles])
            evaluations += self.population_size

            for i in range(self.population_size):
                if new_fitness[i] < local_best_fitness[i]:
                    local_best_fitness[i] = new_fitness[i]
                    self.local_best[i] = self.particles[i]
                if new_fitness[i] < global_best_fitness:
                    global_best_fitness = new_fitness[i]
                    self.global_best = self.particles[i]

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.particles[a] + self.mutation_factor * (self.particles[b] - self.particles[c])
                mutant = np.clip(mutant, lb, ub)
                crossover = np.random.rand(self.dim) < self.recombination_rate
                trial_vector = np.where(crossover, mutant, self.particles[i])

                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < new_fitness[i]:
                    self.particles[i] = trial_vector
                    new_fitness[i] = trial_fitness
                    if trial_fitness < local_best_fitness[i]:
                        local_best_fitness[i] = trial_fitness
                        self.local_best[i] = trial_vector
                    if trial_fitness < global_best_fitness:
                        global_best_fitness = trial_fitness
                        self.global_best = trial_vector

        return self.global_best