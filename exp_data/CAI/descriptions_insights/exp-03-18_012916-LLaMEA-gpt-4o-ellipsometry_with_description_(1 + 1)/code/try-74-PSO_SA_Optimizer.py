import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = max(10, 20 - int(0.1 * budget))
        self.inertia_weight = 0.9  # Adjusted initial inertia weight
        self.c1 = 1.8
        self.c2 = 1.4
        self.temperature = 900

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        evaluations = self.num_particles
        
        while evaluations < self.budget:
            self.inertia_weight = 0.4 + 0.5 * (1 - evaluations / self.budget)  # Dynamic inertia weight
            for i in range(self.num_particles):
                distance_to_best = np.linalg.norm(global_best_position - positions[i])
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +  # Use dynamic inertia weight
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -0.5 * (ub - lb), 0.5 * (ub - lb))  # Adaptive velocity clamping
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                
                F = 0.8 * (1 - evaluations / self.budget)  # Adaptive scaling factor
                a, b, c = np.random.choice(self.num_particles, 3, replace=False)
                mutant_vector = positions[a] + F * (positions[b] - positions[c])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                # Competitive selection between current position and mutant
                if func(mutant_vector) < func(positions[i]):
                    positions[i] = mutant_vector
                
                score = func(positions[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score
                
                if personal_best_scores[i] < global_best_score:
                    global_best_position = personal_best_positions[i]
                    global_best_score = personal_best_scores[i]

            self.temperature *= 0.95 + 0.05 * (global_best_score / max(personal_best_scores))

        return global_best_position, global_best_score