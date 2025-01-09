import numpy as np

class PSO_DE_Metaheuristic:
    def __init__(self, budget, dim, n_particles=30, pso_w=0.7, pso_c1=1.5, pso_c2=1.5, de_cr=0.9, de_f=0.8):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.pso_w = pso_w
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.de_cr = de_cr
        self.de_f = de_f

    def __call__(self, func):
        def within_bounds(position):
            return np.clip(position, -5.0, 5.0)

        def evaluate_fitness(population):
            return np.array([func(ind) for ind in population])

        population = np.random.uniform(-5.0, 5.0, size=(self.n_particles, self.dim))
        velocities = np.zeros((self.n_particles, self.dim))

        best_global_position = population[np.argmin(evaluate_fitness(population))]
        
        for _ in range(self.budget):
            for i in range(self.n_particles):
                pso_r1, pso_r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                de_r1, de_r2, de_r3 = np.random.choice(self.n_particles, 3, replace=False)
                
                velocities[i] = self.pso_w * velocities[i] + self.pso_c1 * pso_r1 * (population[i] - population[i]) + self.pso_c2 * pso_r2 * (best_global_position - population[i])
                new_position = within_bounds(population[i] + velocities[i])
                
                if func(new_position) < func(population[i]):
                    population[i] = new_position
                
                mutant = population[de_r1] + self.de_f * (population[de_r2] - population[de_r3])
                crossover = np.random.rand(self.dim) < self.de_cr
                trial_vector = np.where(crossover, mutant, population[i])
                
                if func(trial_vector) < func(population[i]):
                    population[i] = within_bounds(trial_vector)
                
                if func(population[i]) < func(best_global_position):
                    best_global_position = population[i]

        return best_global_position