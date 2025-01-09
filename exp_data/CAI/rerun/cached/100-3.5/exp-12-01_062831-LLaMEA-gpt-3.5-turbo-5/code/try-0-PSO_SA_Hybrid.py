import numpy as np

class PSO_SA_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def fitness_function(x):
            return func(x)

        def simulated_annealing(x0, T_max=1.0, T_min=0.0001, alpha=0.95, max_iter=100):
            def acceptance_probability(delta_e, temperature):
                if delta_e < 0:
                    return 1.0
                return np.exp(-delta_e / temperature)

            current_state = x0
            best_state = current_state
            for t in range(max_iter):
                T = T_max * (alpha ** t)
                new_state = current_state + np.random.uniform(-0.1, 0.1, size=self.dim)
                delta_e = fitness_function(new_state) - fitness_function(current_state)
                if acceptance_probability(delta_e, T) > np.random.rand():
                    current_state = new_state
                if fitness_function(new_state) < fitness_function(best_state):
                    best_state = new_state
            return best_state

        def pso_sa_hybrid():
            n_particles = 10
            max_iter = self.budget // n_particles
            bounds = (-5.0, 5.0)
            swarm = np.random.uniform(bounds[0], bounds[1], size=(n_particles, self.dim))
            velocities = np.zeros((n_particles, self.dim))

            global_best = swarm[np.argmin([fitness_function(p) for p in swarm])]
            for _ in range(max_iter):
                for i in range(n_particles):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = 0.5 * velocities[i] + 2.0 * r1 * (global_best - swarm[i]) + 2.0 * r2 * (simulated_annealing(swarm[i]) - swarm[i])
                    swarm[i] = np.clip(swarm[i] + velocities[i], bounds[0], bounds[1])
                    if fitness_function(swarm[i]) < fitness_function(global_best):
                        global_best = swarm[i]

            return global_best

        return pso_sa_hybrid()