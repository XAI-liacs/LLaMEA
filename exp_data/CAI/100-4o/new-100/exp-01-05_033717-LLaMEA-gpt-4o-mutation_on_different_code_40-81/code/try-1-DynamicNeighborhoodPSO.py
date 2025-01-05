import numpy as np

class DynamicNeighborhoodPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = 50
        self.w_initial = 0.9  # initial inertia weight
        self.w_final = 0.4   # final inertia weight
        self.c1 = 2.0  # cognitive (particle) coefficient
        self.c2 = 2.0  # social (swarm) coefficient
        self.convergence_threshold = 1e-8
        self.last_improvement = 0

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break
                f = func(positions[i])
                evaluations += 1
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = positions[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = positions[i]
                    self.last_improvement = evaluations

            if evaluations - self.last_improvement > self.budget * 0.1:
                break
                
            neighborhood_size = max(1, int(self.num_particles * (1 - evaluations / self.budget)))
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)

            for i in range(self.num_particles):
                neighbors = np.random.choice(self.num_particles, neighborhood_size, replace=False)
                neighborhood_best = min(neighbors, key=lambda j: personal_best_scores[j])
                velocities[i] = (w * velocities[i] + 
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * np.random.rand() * (personal_best_positions[neighborhood_best] - positions[i]))
                step = self.levy_flight(self.dim)
                positions[i] += velocities[i] + step
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

        return self.f_opt, self.x_opt