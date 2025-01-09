import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
    
    def __call__(self, func):
        np.random.seed(42)
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = swarm.copy()
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_score = np.inf
        global_best_position = None
        
        evaluations = 0
        
        while evaluations < self.budget:
            # Evaluate swarm
            for i in range(self.population_size):
                score = func(swarm[i])
                evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i].copy()
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i].copy()
                
                if evaluations >= self.budget:
                    break

            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(2)
            for i in range(self.population_size):
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best_position - swarm[i]))
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = swarm[indices]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate,
                                        mutant_vector, swarm[i])
                trial_score = func(trial_vector)
                evaluations += 1
                
                # Selection
                if trial_score < personal_best_scores[i]:
                    swarm[i] = trial_vector
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                    
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector

        return global_best_position, global_best_score