class AQPSO:
    # ... [rest of the code remains unchanged]

    def __call__(self, func):
        # ... [rest of the code remains unchanged]
        
        for _ in range(self.budget):
            # ... [rest of the code remains unchanged]
            
            # Update velocities and particles
            r1 = np.random.uniform(size=(self.population_size, self.dim))
            r2 = np.random.uniform(size=(self.population_size, self.dim))
            self.velocities = (
                self.w * self.velocities +
                self.c1 * (1 + self.eval_count / self.budget) * r1 * (self.best_positions - self.particles) +
                self.c2 * (0.5 + self.eval_count / (2 * self.budget)) * r2 * (self.best_global_position - self.particles)
            )
            self.particles += 0.95 * self.velocities  # Adjusted line

            # ... [rest of the code remains unchanged]

        return self.best_global_position, self.best_global_score