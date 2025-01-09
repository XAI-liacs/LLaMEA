import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.vel_max = 0.1 * (self.upper_bound - self.lower_bound)
        self.func_evals = 0

    def __call__(self, func):
        # Initialize the swarm
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        # Evaluate initial positions
        scores = np.array([func(pos) for pos in positions])
        self.func_evals += self.swarm_size
        personal_best_positions = np.copy(positions)
        personal_best_scores = scores
        
        # Find the global best
        g_best_pos = personal_best_positions[np.argmin(personal_best_scores)]
        g_best_score = np.min(personal_best_scores)
        
        # PSO and DE parameters
        inertia_weight = 0.7
        cognitive_param = 1.5
        social_param = 1.5
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        
        while self.func_evals < self.budget:
            # Update velocities and positions for PSO
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            velocities = (inertia_weight * velocities + 
                          cognitive_param * r1 * (personal_best_positions - positions) +
                          social_param * r2 * (g_best_pos - positions))
            velocities = np.clip(velocities, -self.vel_max, self.vel_max)
            positions = positions + velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
            
            # DE mutation and crossover
            for i in range(self.swarm_size):
                candidates = list(range(self.swarm_size))
                candidates.remove(i)
                a, b, c = positions[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, positions[i])
                
                trial_score = func(trial)
                self.func_evals += 1
                
                if trial_score < scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    positions[i] = trial
                    scores[i] = trial_score
                    
                    if trial_score < g_best_score:
                        g_best_pos = trial
                        g_best_score = trial_score

        return g_best_pos, g_best_score