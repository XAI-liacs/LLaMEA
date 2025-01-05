import numpy as np

class QuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5
        self.eta = 0.01

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = lb + (ub - lb) * np.random.rand(self.swarm_size, self.dim)
        velocities = np.random.rand(self.swarm_size, self.dim) * (ub - lb) * 0.1
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(x) for x in swarm])
        global_best_position = swarm[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.w * velocities[i] + \
                                self.c1 * r1 * (personal_best_positions[i] - swarm[i]) + \
                                self.c2 * r2 * (global_best_position - swarm[i])
                
                quantum_displacement = np.random.normal(0, self.eta, self.dim)
                swarm[i] = swarm[i] + velocities[i] + quantum_displacement
                swarm[i] = np.clip(swarm[i], lb, ub)
                
                score = func(swarm[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]
                    
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]
        
        return global_best_position