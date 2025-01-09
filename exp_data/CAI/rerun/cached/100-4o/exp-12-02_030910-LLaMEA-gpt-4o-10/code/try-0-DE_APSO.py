import numpy as np

class DE_APSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.f = 0.5  # DE scaling factor
        self.cr = 0.9  # DE crossover rate
        self.w = 0.5  # PSO inertia weight
        self.c1 = 1.5  # PSO cognitive coefficient
        self.c2 = 1.5  # PSO social coefficient
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        np.random.seed(42)
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        pbest = pop.copy()
        gbest = pop[np.argmin(fitness)]
        pbest_fitness = fitness.copy()
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Differential Evolution/Montage
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Evaluate
                f_trial = func(trial)
                eval_count += 1
                
                # Selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < pbest_fitness[i]:
                        pbest[i] = trial
                        pbest_fitness[i] = f_trial
                        if f_trial < func(gbest):
                            gbest = trial
                
                # Particle Swarm Optimization
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = self.w * velocity[i] + self.c1 * r1 * (pbest[i] - pop[i]) + self.c2 * r2 * (gbest - pop[i])
                pop[i] = np.clip(pop[i] + velocity[i], self.bounds[0], self.bounds[1])
                
                # Evaluate updated position
                f_particles = func(pop[i])
                eval_count += 1
                
                if f_particles < fitness[i]:
                    fitness[i] = f_particles
                    if f_particles < pbest_fitness[i]:
                        pbest[i] = pop[i]
                        pbest_fitness[i] = f_particles
                        if f_particles < func(gbest):
                            gbest = pop[i]
            
            if eval_count >= self.budget:
                break
        
        return gbest