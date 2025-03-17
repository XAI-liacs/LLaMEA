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
        self.alpha = 0.9  # cooling rate

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

        evaluations = 0
        
        while evaluations < self.budget:
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
            
            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = ((0.9 - 0.4 * evaluations / self.budget) * velocities[i] +  # Changed line
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

            # Apply Simulated Annealing
            for i in range(self.swarm_size):
                candidate = positions[i] + np.random.normal(0, 0.1, self.dim)
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                evaluations += 1

                if candidate_score < personal_best_scores[i] or \
                   np.random.rand() < np.exp(-(candidate_score - personal_best_scores[i]) / self.temperature):
                    personal_best_positions[i] = candidate
                    personal_best_scores[i] = candidate_score

                if evaluations >= self.budget:
                    break

            # Cool down the temperature
            self.temperature *= self.alpha

        return global_best_position