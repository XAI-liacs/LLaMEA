import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_search(swarm_size, max_iter):
            # PSO initialization
            swarm = np.random.uniform(-5.0, 5.0, (swarm_size, self.dim))
            best_swarm_pos = swarm[np.argmin([func(p) for p in swarm])]
            best_swarm_val = func(best_swarm_pos)

            for _ in range(max_iter):
                for i in range(swarm_size):
                    new_pos = swarm[i] + np.random.uniform(-1, 1, self.dim) * (best_swarm_pos - swarm[i])
                    new_pos = np.clip(new_pos, -5.0, 5.0)
                    new_val = func(new_pos)
                    if new_val < func(swarm[i]):
                        swarm[i] = new_pos
                        if new_val < best_swarm_val:
                            best_swarm_pos = new_pos
                            best_swarm_val = new_val
            return best_swarm_pos

        def sa_search(init_temp, final_temp, cooling_rate):
            # Simulated Annealing initialization
            current_pos = np.random.uniform(-5.0, 5.0, self.dim)
            current_val = func(current_pos)
            best_pos = current_pos
            best_val = current_val
            temperature = init_temp

            while temperature > final_temp:
                new_pos = current_pos + np.random.uniform(-1, 1, self.dim)
                new_pos = np.clip(new_pos, -5.0, 5.0)
                new_val = func(new_pos)
                if new_val < current_val or np.random.rand() < np.exp((current_val - new_val) / temperature):
                    current_pos = new_pos
                    current_val = new_val
                    if new_val < best_val:
                        best_pos = new_pos
                        best_val = new_val
                temperature *= cooling_rate
            return best_pos

        # Improved Hybrid PSO-SA optimization
        pso_ratio = 0.2  # Initial PSO exploitation ratio
        for _ in range(self.budget):
            best_pos = pso_search(30, int(1000 * pso_ratio))  # PSO exploration phase with dynamic iterations
            best_pos = sa_search(100, 0.1, 0.95)  # SA exploitation phase
            pso_ratio *= 0.95  # Adjust PSO exploitation ratio dynamically

        return best_pos