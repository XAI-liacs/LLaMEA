import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=20, alpha=0.9, beta=0.999):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        def pso_sa(func, budget, dim, swarm_size, alpha, beta):
            def objective_function(x):
                return func(x)

            def initialize_swarm(dim, swarm_size):
                return np.random.uniform(-5.0, 5.0, (swarm_size, dim)), np.full(swarm_size, np.inf)

            swarm, best_position = initialize_swarm(dim, swarm_size)
            best_value = np.inf

            for _ in range(budget):
                for i in range(swarm_size):
                    candidate = swarm[i] + alpha * np.random.normal(0, 1, dim)
                    candidate = np.clip(candidate, -5.0, 5.0)
                    candidate_value = objective_function(candidate)
                    if candidate_value < best_value:
                        best_value = candidate_value
                        best_position = candidate
                    if candidate_value < best_position[i]:
                        swarm[i] = candidate
                        best_position[i] = candidate_value
                    else:
                        delta = candidate_value - best_position[i]
                        acceptance_prob = np.exp(-delta / beta)
                        if np.random.rand() < acceptance_prob:
                            swarm[i] = candidate
                            best_position[i] = candidate_value

            return best_position

        return pso_sa(func, self.budget, self.dim, self.swarm_size, self.alpha, self.beta)