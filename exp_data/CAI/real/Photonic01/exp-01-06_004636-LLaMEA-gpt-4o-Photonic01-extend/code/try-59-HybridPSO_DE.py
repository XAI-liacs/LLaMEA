import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7   # inertia weight
        self.f = 0.5   # scaling factor for differential evolution
        self.cr = 0.9  # crossover probability
        self.velocities = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        pbests = pop.copy()
        pbest_scores = np.array([func(ind) for ind in pbests])
        gbest = pbests[np.argmin(pbest_scores)].copy()
        gbest_score = np.min(pbest_scores)

        evals = self.population_size
        
        while evals < self.budget:
            self.c1 = 1.5 * (0.6 + 0.4 * (1 - evals / self.budget))  # Changed line for enhanced decay
            self.c2 = 1.5 * (0.6 + 0.4 * (evals / self.budget))  # Added line for social coefficient decay
            self.w = 0.5 + 0.4 * (1 - evals / self.budget)  # Changed line for improved inertia weight
            if evals % self.population_size < self.population_size // 2:
                # Particle Swarm Optimization
                r1, r2 = np.random.rand(2, self.population_size, self.dim)
                self.velocities = (self.w - 0.4 * (1 - (evals / self.budget)**2)) * self.velocities + self.c1 * r1 * (pbests - pop) + self.c2 * r2 * (gbest - pop)
                pop = np.clip(pop + self.velocities, lb, ub)
            else:
                # Differential Evolution
                self.cr = 0.7 + 0.3 * (1 - evals / self.budget)  # Changed line for adaptive crossover probability
                for i in range(self.population_size):
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.f * (b - c), lb, ub)
                    cross_points = np.random.rand(self.dim) < self.cr
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    pop[i] = np.where(cross_points, mutant, pop[i])

            # Evaluate the population
            scores = np.array([func(ind) for ind in pop])
            evals += self.population_size

            # Update personal bests
            better_mask = scores < pbest_scores
            pbests[better_mask] = pop[better_mask]
            pbest_scores[better_mask] = scores[better_mask]

            # Update global best
            if np.min(pbest_scores) < gbest_score:
                gbest = pbests[np.argmin(pbest_scores)].copy()
                gbest_score = np.min(pbest_scores)

        return gbest, gbest_score