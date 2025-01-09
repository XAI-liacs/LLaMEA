import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia=0.7, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particle positions and velocities
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_index = np.argmin(personal_best_scores)
        
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        
        evaluations = self.swarm_size
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social * r2 * (global_best_position - positions[i]))
                positions[i] = positions[i] + velocities[i]
                np.clip(positions[i], lb, ub, out=positions[i])
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
            
            self.f_opt, self.x_opt = global_best_score, global_best_position

        return self.f_opt, self.x_opt