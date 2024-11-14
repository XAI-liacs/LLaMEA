import numpy as np

class DynamicMutationAdaptiveStepSizeEnhancedSocialSwarmOptimization(DynamicPopulationSizeEnhancedSocialSwarmOptimization):
    def __call__(self, func):
        def _initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def _local_search(x, f):
            step_size = 1.0
            for _ in range(10):
                x_new = x + step_size * np.random.normal(size=self.dim)
                if f(x_new) < f(x):
                    x = x_new
                    step_size *= 1.1  # Adaptive step adjustment
                else:
                    step_size *= 0.9  # Adaptive step adjustment
            return x

        swarm = _initialize_swarm()
        best_position = swarm[np.argmin([func(x) for x in swarm])]
        for _ in range(self.budget):
            diversity = np.mean(np.std(swarm, axis=0))
            new_swarm_size = max(5, int(self.swarm_size * (1 - diversity / 10.0)))  # Dynamic population size adjustment
            selected_indices = np.random.choice(range(self.swarm_size), new_swarm_size, replace=False)
            swarm = swarm[selected_indices]
            self.swarm_size = new_swarm_size
            for i in range(self.swarm_size):
                r_p = np.random.uniform(0, 1, size=self.dim)
                r_g = np.random.uniform(0, 1)
                step_size = 1.0  # Reset step size for dynamic mutation
                for _ in range(5):  # Apply dynamic mutation for only 5 iterations
                    swarm[i] = self.omega * swarm[i] + self.phi_p * r_p * (best_position - swarm[i]) + self.phi_g * r_g * (best_position - swarm[i])
                    swarm[i] = _local_search(swarm[i], func)
                    if func(swarm[i]) < func(best_position):
                        best_position = np.copy(swarm[i])
                        step_size *= 1.1  # Adaptive step adjustment
                    else:
                        step_size *= 0.9  # Adaptive step adjustment
        return best_position