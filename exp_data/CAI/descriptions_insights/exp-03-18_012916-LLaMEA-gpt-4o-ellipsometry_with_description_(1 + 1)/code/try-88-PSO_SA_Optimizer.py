import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = max(10, 20 - int(0.1 * budget))
        self.inertia_weight = 0.9
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
            diversity = np.mean(np.std(positions, axis=0))  # Calculate swarm diversity
            self.inertia_weight = 0.4 + 0.5 * (1 - evaluations / self.budget)
            adaptive_learning_rate = 0.5 + 0.5 * (diversity / (ub - lb).mean())  # Adaptive learning rate
            
            for i in range(self.num_particles):
                distance_to_best = np.linalg.norm(global_best_position - positions[i])
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                velocity_scaling = 1 - 0.5 * (diversity / (ub - lb).mean())  # New scaling factor based on diversity
                velocities[i] = (self.inertia_weight * velocities[i] + 
                                 adaptive_learning_rate * self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 adaptive_learning_rate * self.c2 * r2 * (global_best_position - positions[i])) * velocity_scaling
                velocities[i] *= np.tanh(1 / (1 + np.exp(-distance_to_best / np.sum(np.linalg.norm(velocities, axis=1)))))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                
                F = 0.8 * (1 - evaluations / self.budget)
                a, b, c = np.random.choice(self.num_particles, 3, replace=False)
                mutant_vector = positions[a] + F * (positions[b] - positions[c])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                score = func(positions[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score
                
                if func(mutant_vector) < score:
                    positions[i] = mutant_vector

                if personal_best_scores[i] < global_best_score:
                    global_best_position = personal_best_positions[i]
                    global_best_score = personal_best_scores[i]

            self.temperature *= 0.95 + 0.05 * (global_best_score / max(personal_best_scores))

        return global_best_position, global_best_score