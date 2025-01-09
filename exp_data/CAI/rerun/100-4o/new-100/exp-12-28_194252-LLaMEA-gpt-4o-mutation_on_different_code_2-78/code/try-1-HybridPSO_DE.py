import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10, swarm_size=50, c1=2.0, c2=2.0, w=0.7):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.f_opt = np.Inf
        self.x_opt = None
        
        self.velocities = np.random.uniform(-1, 1, (swarm_size, dim))
        self.positions = np.random.uniform(-5.0, 5.0, (swarm_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(swarm_size, np.inf)
        
    def __call__(self, func):
        global_best_score = np.inf
        global_best_position = None
        
        evaluations = 0
        while evaluations < self.budget:
            self.w = 0.4 + 0.3 * ((self.budget - evaluations) / self.budget)  # Adaptive inertia weight
            for i in range(self.swarm_size):
                f = func(self.positions[i])
                evaluations += 1
                
                if f < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = f
                    self.personal_best_positions[i] = self.positions[i].copy()
                    
                if f < global_best_score:
                    global_best_score = f
                    global_best_position = self.positions[i].copy()
                    self.f_opt, self.x_opt = global_best_score, global_best_position

                if evaluations >= self.budget:
                    break
            
            # Velocity and position update using modified PSO+DE approach
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] + 
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + 
                                      self.c2 * r2 * (global_best_position - self.positions[i]))
                
                # Differential Evolution Mutation
                idxs = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                mutant_vector = self.positions[idxs[0]] + 0.5 * (self.positions[idxs[1]] - self.positions[idxs[2]])
                
                # Crossover using fitness-based probability
                crossover_mask = np.random.rand(self.dim) < (0.9 if self.personal_best_scores[i] < np.median(self.personal_best_scores) else 0.5)
                trial_vector = np.where(crossover_mask, mutant_vector, self.positions[i] + self.velocities[i])
                trial_vector = np.clip(trial_vector, -5.0, 5.0)
                
                # Accept new position if better
                f_trial = func(trial_vector)
                evaluations += 1
                if f_trial < self.personal_best_scores[i]:
                    self.positions[i] = trial_vector
                    self.personal_best_scores[i] = f_trial
                    self.personal_best_positions[i] = trial_vector
                    if f_trial < global_best_score:
                        global_best_score = f_trial
                        global_best_position = trial_vector
                        self.f_opt, self.x_opt = global_best_score, global_best_position

                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt