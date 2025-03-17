import numpy as np

class PSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) - 0.5
        self.personal_best = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_score = np.inf
        self.iteration = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        scale = ub - lb
        self.particles = lb + self.particles * scale
        cognitive_constant = 1.5
        social_constant = 1.5
        temperature = 1.0
        temperature_decay = 0.99

        while self.iteration < self.budget:
            inertia_weight = 0.9 - 0.5 * (self.iteration / self.budget)
            scores = np.array([func(p) for p in self.particles])
            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best[i] = self.particles[i]

            best_particle_idx = np.argmin(scores)
            if scores[best_particle_idx] < self.global_best_score:
                self.global_best_score = scores[best_particle_idx]
                self.global_best = self.particles[best_particle_idx]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (
                    inertia_weight * self.velocities[i] +
                    cognitive_constant * r1 * (self.personal_best[i] - self.particles[i]) +
                    social_constant * r2 * (self.global_best - self.particles[i])
                )
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lb, ub)

                # Simulated annealing step
                proposed_solution = self.particles[i] + np.random.normal(0, 0.1, self.dim) * scale
                proposed_solution = np.clip(proposed_solution, lb, ub)
                proposed_score = func(proposed_solution)

                if proposed_score < scores[i] or np.random.rand() < np.exp((scores[i] - proposed_score) / temperature):
                    self.particles[i] = proposed_solution
                    scores[i] = proposed_score

                # Differential Evolution Mutation Step
                if np.random.rand() < 0.2:
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + 0.8 * (b - c), lb, ub)
                    mutant_score = func(mutant)
                    if mutant_score < scores[i]:
                        self.particles[i] = mutant
                        scores[i] = mutant_score

            temperature *= temperature_decay
            self.iteration += self.population_size

        return self.global_best, self.global_best_score