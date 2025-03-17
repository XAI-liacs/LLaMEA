import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.w = 0.7  
        self.c1 = 1.5  
        self.c2 = 1.5  
        self.temperature = 1000
        self.alpha = 0.95  
        self.beta = 1.5  # Lévy flight parameter

    def levy_flight(self):
        return np.random.standard_cauchy(self.dim) * 0.01

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        v_max = (ub - lb) * 0.1
        
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
            
            w_adjusted = self.w * (0.5 + np.random.rand() / 2)  # Changed line for adaptive inertia weight
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w_adjusted * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                positions[i] += velocities[i] + self.levy_flight()
                positions[i] = np.clip(positions[i], lb, ub)

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

            self.temperature *= self.alpha
            
            if evaluations % (self.budget // 10) == 0:
                gradient_mutation = np.gradient(global_best_position) * np.random.rand(self.dim) * 0.01  # Added line for gradient-based mutation
                global_best_position += gradient_mutation  # Changed line for applying the mutation

        return global_best_position