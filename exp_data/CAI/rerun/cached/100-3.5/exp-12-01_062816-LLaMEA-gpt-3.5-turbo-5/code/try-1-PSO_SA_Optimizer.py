import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_sa_optimization():
            # PSO parameters
            w = 0.5
            c1 = 1.5
            c2 = 1.5
            pso_iterations = 50

            # SA parameters
            t_initial = 100.0
            t_final = 1.0
            sa_iterations = 50

            # Initialize particles
            particles = np.random.uniform(-5.0, 5.0, size=(self.dim,))
            best_particle = particles.copy()
            best_fitness = func(best_particle)

            temperature = t_initial

            for _ in range(self.budget // (pso_iterations + sa_iterations)):
                # PSO phase
                for _ in range(pso_iterations):
                    r1 = np.random.uniform(0, 1, size=(self.dim,))
                    r2 = np.random.uniform(0, 1, size=(self.dim,))
                    velocities = w * velocities + c1 * r1 * (best_particle - particles) + c2 * r2 * (best_particle - particles)
                    particles = particles + velocities
                    particles = np.clip(particles, -5.0, 5.0)
                    fitness = func(particles)
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_particle = particles.copy()

                # SA phase
                for _ in range(sa_iterations):
                    candidate_particle = best_particle + np.random.normal(0, 1, size=(self.dim,)) * temperature
                    candidate_particle = np.clip(candidate_particle, -5.0, 5.0)
                    candidate_fitness = func(candidate_particle)
                    delta_e = candidate_fitness - best_fitness
                    if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temperature):
                        best_particle = candidate_particle
                        best_fitness = candidate_fitness

                    temperature = t_initial + (_ / sa_iterations) * (t_final - t_initial)

            return best_particle

        return pso_sa_optimization()