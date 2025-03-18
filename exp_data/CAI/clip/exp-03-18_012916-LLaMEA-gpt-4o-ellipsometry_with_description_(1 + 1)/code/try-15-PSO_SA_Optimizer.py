import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.temperature = 1000
        
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
            for i in range(self.num_particles):
                distance_to_best = np.linalg.norm(global_best_position - positions[i]) 
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                velocities[i] = ((0.4 + (self.budget - evaluations) / self.budget * 0.3) * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                velocities[i] *= np.exp(-distance_to_best / np.sum(np.linalg.norm(velocities, axis=1)))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                
                score = func(positions[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score
                
                # Opposition-based learning for better exploration
                opposite_positions = lb + ub - positions[i]
                opposite_positions = np.clip(opposite_positions, lb, ub)
                opposite_score = func(opposite_positions)
                evaluations += 1
                
                if opposite_score < score or np.random.rand() < np.exp((score - opposite_score) / self.temperature):
                    positions[i] = opposite_positions
                
                # Adaptive mutation based on diversity
                diversity = np.mean(np.linalg.norm(personal_best_positions - global_best_position, axis=1))
                mutation_strength = 0.1 * (1 - diversity / (ub - lb).mean())
                perturbed_position = positions[i] + np.random.normal(0, mutation_strength, self.dim)
                perturbed_position = np.clip(perturbed_position, lb, ub)
                perturbed_score = func(perturbed_position)
                evaluations += 1

                if perturbed_score < score or np.random.rand() < np.exp((score - perturbed_score) / self.temperature):
                    positions[i] = perturbed_position

                if personal_best_scores[i] < global_best_score:
                    global_best_position = personal_best_positions[i]
                    global_best_score = personal_best_scores[i]

            self.temperature *= 0.99

        return global_best_position, global_best_score