import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 50
        self.num_individuals = 10
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.best_particle_positions = np.copy(self.particles)
        self.best_global_position = np.copy(self.particles[0])
        self.best_particle_costs = np.full(self.num_particles, np.inf)
        self.best_global_cost = np.inf
        self.individuals = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_individuals, self.dim))
        self.CR = 0.9
        self.F = 0.8

    def __call__(self, func):
        eval_count = 0

        while eval_count < self.budget:
            # Evaluate particles
            for i in range(self.num_particles):
                cost = func(self.particles[i])
                eval_count += 1
                if cost < self.best_particle_costs[i]:
                    self.best_particle_costs[i] = cost
                    self.best_particle_positions[i] = np.copy(self.particles[i])
                if cost < self.best_global_cost:
                    self.best_global_cost = cost
                    self.best_global_position = np.copy(self.particles[i])

            if eval_count >= self.budget:
                break

            # Dynamic Inertia PSO update
            inertia_weight = 0.4 + 0.1 * (1 - eval_count / self.budget)
            cognitive_component = np.random.rand(self.num_particles, self.dim)
            social_component = np.random.rand(self.num_particles, self.dim)

            self.velocities = (inertia_weight * self.velocities
                               + cognitive_component * (self.best_particle_positions - self.particles)
                               + social_component * (self.best_global_position - self.particles))
            self.particles += self.velocities
            self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

            # Targeted DE Mutation and Crossover
            for j in range(self.num_individuals):
                if eval_count >= self.budget:
                    break
                indices = list(range(self.num_individuals))
                indices.remove(j)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.individuals[a] + self.F * (self.individuals[b] - self.individuals[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.individuals[j])
                trial_cost = func(trial)
                eval_count += 1
                if trial_cost < func(self.individuals[j]):
                    self.individuals[j] = trial

            # Adaptive DE adjustment
            if eval_count % (self.budget // 10) == 0:
                self.CR = np.clip(self.CR + 0.1 * (np.random.rand() - 0.5), 0, 1)
                self.F = np.clip(self.F + 0.1 * (np.random.rand() - 0.5), 0.5, 1.0)

        return self.best_global_position, self.best_global_cost