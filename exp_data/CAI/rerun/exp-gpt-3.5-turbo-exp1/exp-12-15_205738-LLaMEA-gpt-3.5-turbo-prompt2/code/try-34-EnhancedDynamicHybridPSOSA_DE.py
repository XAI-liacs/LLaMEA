class EnhancedDynamicHybridPSOSA_DE(ImprovedDynamicHybridPSOSA_DE):
    def dynamic_mutation(self, fitness_history):
        self.f = max(self.f_lower, min(self.f_upper, np.mean(fitness_history)))  # Dynamic F based on mean fitness
        self.cr = max(self.cr_lower, min(self.cr_upper, np.std(fitness_history)))  # Dynamic CR based on fitness standard deviation

    def __call__(self, func):
        lb, ub = -5.0, 5.0
        swarm = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        velocity = np.zeros((self.num_particles, self.dim))
        personal_best_pos = swarm.copy()
        personal_best_val = np.full(self.num_particles, np.inf)
        global_best_pos = np.zeros(self.dim)
        global_best_val = np.inf

        curr_temp = self.initial_temp
        fitness_history = []

        for _ in range(self.max_iterations):
            for i in range(self.num_particles):
                fitness = func(swarm[i])
                fitness_history.append(fitness)

                if fitness < personal_best_val[i]:
                    personal_best_val[i] = fitness
                    personal_best_pos[i] = swarm[i].copy()
                    if fitness < global_best_val:
                        global_best_val = fitness
                        global_best_pos = swarm[i].copy()

                r1, r2 = np.random.rand(), np.random.rand()
                velocity[i] = 0.5 * velocity[i] + self.c1 * r1 * (personal_best_pos[i] - swarm[i]) + self.c2 * r2 * (global_best_pos - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocity[i], lb, ub)

                a, b, c = swarm[np.random.choice(range(self.num_particles), 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = swarm[i].copy()
                trial[crossover] = mutant[crossover]
                trial_fitness = func(trial)

                if trial_fitness < fitness:
                    swarm[i] = trial.copy()

            curr_temp *= self.cooling_rate
            self.dynamic_mutation(fitness_history)

        return global_best_val