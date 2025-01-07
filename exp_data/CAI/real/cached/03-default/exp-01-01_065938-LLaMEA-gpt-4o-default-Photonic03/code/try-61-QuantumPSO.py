import numpy as np

class QuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.quantum_factor = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        lb, ub = np.array(lb), np.array(ub)

        pos = lb + np.random.rand(self.swarm_size, self.dim) * (ub - lb)
        vel = np.random.rand(self.swarm_size, self.dim) * 0.1 * (ub - lb)
        pbest_pos = np.copy(pos)
        pbest_val = np.array([func(x) for x in pos])
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = np.copy(pbest_pos[gbest_idx])

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Quantum-inspired update
                quantum_position = np.random.rand(self.dim) * self.quantum_factor
                vel[i] = (self.w * vel[i] 
                          + self.c1 * r1 * (pbest_pos[i] - pos[i]) 
                          + self.c2 * r2 * (gbest_pos - pos[i])) + quantum_position
                
                pos[i] = np.clip(pos[i] + vel[i], lb, ub)
                f_value = func(pos[i])
                evaluations += 1

                if f_value < pbest_val[i]:
                    pbest_pos[i] = np.copy(pos[i])
                    pbest_val[i] = f_value
                    if f_value < pbest_val[gbest_idx]:
                        gbest_idx = i
                        gbest_pos = np.copy(pbest_pos[gbest_idx])

            self.w = np.clip(self.w * 0.99 + 0.01 * np.random.rand(), 0.4, 0.9)
            self.c1 = np.clip(self.c1 * 0.99 + 0.01 * np.random.rand(), 1.4, 2.0)
            self.c2 = np.clip(self.c2 * 0.99 + 0.01 * np.random.rand(), 1.4, 2.0)
            self.quantum_factor = np.clip(self.quantum_factor * 0.95 + 0.05 * np.random.rand(), 0.01, 0.2)

        return gbest_pos