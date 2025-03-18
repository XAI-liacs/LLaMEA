# In the function `detect_periodic_pattern`, modify the threshold from 1.1 to 1.15
class SwarmPeriodicOptimizer:
    def detect_periodic_pattern(self, sequence):
        length = len(sequence)
        autocorrelation = np.correlate(sequence, sequence, mode='full')
        autocorrelation = autocorrelation[length-1:]
        peaks = np.where((autocorrelation[1:] < autocorrelation[:-1]) & 
                         (autocorrelation[:-1] > np.mean(autocorrelation) * 1.15))[0]  # Adjusted threshold from 1.1 to 1.15
        if peaks.size > 0:
            period = peaks[0] + 1
            return sequence[:period]
        return None