import numpy as np

class Enhanced_HPSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w_initial = 0.9
        self.w_final = 0.4
        self.F = 0.8   # differential weight
        self.CR = 0.9  # crossover probability
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.pbest_position = self.position.copy()
        self.pbest_value = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_value = np.inf
        self.eval_count = 0
    
    def evaluate(self, func, position):
        if self.eval_count < self.budget:
            value = func(position)
            self.eval_count += 1
            return value
        else:
            raise Exception("Budget exceeded")
        
    def __call__(self, func):
        for i in range(self.population_size):
            fitness = self.evaluate(func, self.position[i])
            if fitness < self.pbest_value[i]:
                self.pbest_value[i] = fitness
                self.pbest_position[i] = self.position[i].copy()
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = self.position[i].copy()
        
        while self.eval_count < self.budget:
            w = self.w_initial - (self.w_initial - self.w_final) * (self.eval_count / self.budget)
            for i in range(self.population_size):
                # PSO Update
                r1, r2 = np.random.rand(2)
                self.velocity[i] = (
                    w * self.velocity[i] +
                    self.c1 * r1 * (self.pbest_position[i] - self.position[i]) +
                    self.c2 * r2 * (self.gbest_position - self.position[i])
                )
                self.position[i] += self.velocity[i]
                self.position[i] = np.clip(self.position[i], self.lower_bound, self.upper_bound)
                
                # DE Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.position[a] + self.F * (self.position[b] - self.position[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # DE Crossover
                trial = np.copy(self.position[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]
                
                # Evaluate trial
                trial_fitness = self.evaluate(func, trial)
                if trial_fitness < self.pbest_value[i]:
                    if trial_fitness < self.evaluate(func, self.position[i]):  # selective update
                        self.pbest_value[i] = trial_fitness
                        self.pbest_position[i] = trial.copy()
                    
                # Update global best if improved
                if trial_fitness < self.gbest_value:
                    self.gbest_value = trial_fitness
                    self.gbest_position = trial.copy()
                    
        return self.gbest_position, self.gbest_value