import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive parameter
        self.c2 = 1.5  # social parameter
        self.temperature = 1000
        self.alpha = 0.95  # Adjusted cooling rate
        self.elite_fraction = 0.2  # Fraction of elite particles
        self.mutation_rate = 0.1  # Initial mutation rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        v_max = (ub - lb) * 0.1
        
        # Initialize the swarm
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-v_max, v_max, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        elite_count = int(self.swarm_size * self.elite_fraction)

        evaluations = 0
        
        while evaluations < self.budget:
            self.swarm_size = max(10, int(30 * (1 - evaluations / self.budget)))  # Adaptive swarm size
            self.w = max(0.1, 0.5 * (1 - evaluations / self.budget))  # Dynamic inertia weight
            self.c1 = max(0.5, 1.5 * (1 - (evaluations / self.budget) ** 2))  # Non-linear cognitive parameter reduction
            for i in range(self.swarm_size):
                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break
            
            # Update velocities and positions with elite learning and random perturbation
            elite_indices = np.argsort(personal_best_scores)[:elite_count]
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                elite_position = personal_best_positions[np.random.choice(elite_indices)]
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (elite_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                positions[i] += velocities[i]
                positions[i] += np.random.normal(0, self.mutation_rate * (1 - evaluations / self.budget), self.dim)
                positions[i] = np.clip(positions[i], lb, ub)

            # Apply Simulated Annealing with adaptive mutation
            self.mutation_rate = 0.05 + 0.95 * (evaluations / self.budget)  # Adjust mutation rate
            elite_count = int(self.swarm_size * self.elite_fraction * (1.2 + evaluations / (2 * self.budget)))  # Adjusted dynamic elite fraction
            for i in range(self.swarm_size):
                candidate = positions[i] + np.random.normal(0, self.mutation_rate, self.dim)
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                evaluations += 1

                if candidate_score < personal_best_scores[i] or \
                   np.random.rand() < np.exp(-(candidate_score - personal_best_scores[i]) / self.temperature):
                    personal_best_positions[i] = candidate
                    personal_best_scores[i] = candidate_score

                if evaluations >= self.budget:
                    break

            # Cool down the temperature non-linearly
            self.temperature *= (self.alpha ** 2)

        return global_best_position