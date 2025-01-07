import numpy as np

class QuantumInspiredMemetic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(size=(self.population_size, dim)) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def chaotic_local_search(self, position, func, lower_bound, upper_bound):
        beta = 0.1
        steps = int(np.sqrt(self.dim))
        best_position = position.copy()
        best_fitness = func(best_position)

        for _ in range(steps):
            perturbation = (np.random.rand(self.dim) - 0.5) * 2 * beta * (upper_bound - lower_bound)
            candidate = np.clip(position + perturbation, lower_bound, upper_bound)
            candidate_fitness = func(candidate)
            self.fitness_evaluations += 1
            if candidate_fitness < best_fitness:
                best_fitness = candidate_fitness
                best_position = candidate.copy()
            beta *= 0.9  # Chaotic decay
            if self.fitness_evaluations >= self.budget:
                break

        return best_position, best_fitness

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                inertia_weight = 0.5 + (0.9 - 0.5) * (1 - self.fitness_evaluations / self.budget)
                cognitive_coeff = 1.4
                social_coeff = 1.6
                r1, r2 = np.random.rand(), np.random.rand()

                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                if np.random.rand() < 0.2:  # Probability for local search
                    local_position, local_fitness = self.chaotic_local_search(self.particles[i], func, lower_bound, upper_bound)
                    if local_fitness < self.personal_best_fitness[i]:
                        self.particles[i] = local_position.copy()
                        self.personal_best[i] = local_position.copy()
                        self.personal_best_fitness[i] = local_fitness
                        if local_fitness < self.global_best_fitness:
                            self.global_best_fitness = local_fitness
                            self.global_best = local_position.copy()

        return self.global_best