import numpy as np

class Optimized_Enhanced_PSO_SA_Optimizer_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_sa_optimization():
            def objective_function(x):
                return func(x)
            
            n_particles = 10
            max_iterations_pso = 100
            max_iterations_sa = 10
            bounds = (-5.0, 5.0)
            alpha = 0.95
            T0 = 1.0
            T_min = 0.0001
            sigma = 0.1
            inertia_max = 0.9
            inertia_min = 0.4

            best_solution = None
            best_fitness = float('inf')

            particles = np.random.uniform(bounds[0], bounds[1], (n_particles, self.dim))
            velocities = np.zeros((n_particles, self.dim))
            inertia_weight = inertia_max

            random_cognitive = np.random.random((n_particles, self.dim))
            random_social = np.random.random((n_particles, self.dim))

            for _ in range(max_iterations_pso):
                for i in range(n_particles):
                    fitness = objective_function(particles[i])
                    if fitness < best_fitness:
                        best_solution = particles[i].copy()
                        best_fitness = fitness

                    cognitive_component = random_cognitive[i] * (particles[i] - particles[i])
                    social_component = random_social[i] * (best_solution - particles[i])
                    cognitive_social = alpha * (cognitive_component + social_component)
                    velocities[i] = inertia_weight * velocities[i] + cognitive_social
                    particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

                inertia_weight = max(inertia_max - (_ / max_iterations_pso) * (inertia_max - inertia_min), inertia_min)

            current_solution = best_solution
            current_fitness = best_fitness
            T = T0

            while T > T_min:
                for _ in range(max_iterations_sa):
                    candidate_solution = np.clip(current_solution + np.random.normal(0, sigma, self.dim), bounds[0], bounds[1])
                    candidate_fitness = objective_function(candidate_solution)

                    if candidate_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - candidate_fitness) / T):
                        current_solution = candidate_solution
                        current_fitness = candidate_fitness

                T *= alpha

            return current_solution

        return pso_sa_optimization()