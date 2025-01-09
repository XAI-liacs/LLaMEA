import numpy as np

class HybridPSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.inertia_weight = 0.5
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.vel_limit = (self.upper_bound - self.lower_bound) / 2
    
    def __call__(self, func):
        np.random.seed(0)
        # Initialize swarm
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.vel_limit, self.vel_limit, (self.population_size, self.dim))
        
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        evals = self.population_size
        
        while evals < self.budget:
            for i in range(self.population_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -self.vel_limit, self.vel_limit)

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate
                current_score = func(positions[i])
                evals += 1
                
                # Update personal and global bests
                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = current_score

                if current_score < global_best_score:
                    global_best_position = positions[i]
                    global_best_score = current_score

                if evals >= self.budget:
                    break
            
            # Adaptive Differential Mutation
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_prob:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = personal_best_positions[idxs[0]], personal_best_positions[idxs[1]], personal_best_positions[idxs[2]]
                    mutant_vector = a + self.mutation_factor * (b - c)
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    
                    trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, positions[i])
                    trial_score = func(trial_vector)
                    evals += 1
                    
                    if trial_score < personal_best_scores[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_scores[i] = trial_score
                        
                    if trial_score < global_best_score:
                        global_best_position = trial_vector
                        global_best_score = trial_score
                        
                    if evals >= self.budget:
                        break

        return global_best_position, global_best_score