class HarmonySearch:
    def __call__(self, func):
        for _ in range(self.budget):
            new_harmony = np.copy(self.memory)
            
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = self.memory[i]
                else:
                    new_harmony[i] = np.random.uniform(-5.0, 5.0)
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-self.bw, self.bw)
                        self.bw *= 0.95  # Adaptive bandwidth control
                
            new_harmony_fitness = func(new_harmony)
            if new_harmony_fitness < self.f_opt:
                self.f_opt = new_harmony_fitness
                self.x_opt = new_harmony
                self.memory = new_harmony
        return self.f_opt, self.x_opt