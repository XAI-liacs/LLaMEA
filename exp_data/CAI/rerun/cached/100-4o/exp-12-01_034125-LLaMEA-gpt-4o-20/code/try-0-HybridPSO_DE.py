import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.f = 0.8  # DE mutation factor
        self.cr = 0.9  # DE crossover probability
        self.bounds = (-5.0, 5.0)
    
    def __call__(self, func):
        # Initialize particle swarm
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        
        # Global best initialization
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_param * r1 * (personal_best_positions[i] - pop[i]) +
                                 self.social_param * r2 * (global_best_position - pop[i]))
                
                # Update position
                pop[i] = np.clip(pop[i] + velocities[i], self.bounds[0], self.bounds[1])
                
                # Evaluate new position
                score = func(pop[i])
                evaluations += 1
                
                # Update personal best if necessary
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = pop[i]
                    
                # Update global best if necessary
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = pop[i]
                
                # Apply Differential Evolution crossover
                if evaluations < self.budget:
                    # Select three random indices different from i
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    
                    mutant = np.clip(a + self.f * (b - c), self.bounds[0], self.bounds[1])
                    trial = np.copy(pop[i])
                    crossover = np.random.rand(self.dim) < self.cr
                    trial[crossover] = mutant[crossover]
                    
                    # Evaluate trial
                    trial_score = func(trial)
                    evaluations += 1
                    
                    # Replace if trial is better
                    if trial_score < score:
                        pop[i] = trial
                        score = trial_score
                
                # Update global best after DE selection as well
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = pop[i]
        
        return global_best_position, global_best_score