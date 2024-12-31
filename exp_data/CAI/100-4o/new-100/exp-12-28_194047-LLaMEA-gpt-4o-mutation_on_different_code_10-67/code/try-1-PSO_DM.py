import numpy as np

class PSO_DM:
    def __init__(self, budget=10000, dim=10, swarm_size=30, w=0.5, c1=1.5, c2=1.5, mutation_factor=0.8):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.mutation_factor = mutation_factor
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.Inf)

        global_best_position = None
        global_best_score = np.Inf

        evaluations = 0

        while evaluations < self.budget:
            # Evaluate current swarm positions
            scores = np.apply_along_axis(func, 1, positions)
            evaluations += self.swarm_size

            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]

            # Update global best
            min_index = np.argmin(personal_best_scores)
            if personal_best_scores[min_index] < global_best_score:
                global_best_score = personal_best_scores[min_index]
                global_best_position = personal_best_positions[min_index]

            # Update velocities and positions
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            self.w = 0.9 - 0.5 * (evaluations / self.budget)  # Dynamic inertia weight
            velocities = (self.w * velocities 
                         + self.c1 * r1 * (personal_best_positions - positions) 
                         + self.c2 * r2 * (global_best_position - positions))
            positions += velocities

            # Constrain positions to bounds
            positions = np.clip(positions, lb, ub)

            # Differential Mutation
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                idxs = np.random.choice(self.swarm_size, 3, replace=False)
                a, b, c = positions[idxs]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, lb, ub)
                
                current_score = func(mutant)
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = mutant
                    evaluations += 1
                    if current_score < global_best_score:
                        global_best_score = current_score
                        global_best_position = mutant
                self.mutation_factor = 0.5 + 0.4 * np.exp(-current_score)  # Adaptive mutation factor

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt