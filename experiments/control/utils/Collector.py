import numpy as np

# A data collector utility class
# makes it easy to grab arbitrary data during the execution of an agent
# data is indexed by a key given by the user at collection time,
# then is aggregated whenever `reset()` is called
def tolerant_mean(arrs):
        lens = [len(i) for i in arrs]
        arr = np.ma.empty((np.max(lens),len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[:len(l),idx] = l
        return arr.mean(axis = -1), arr.std(axis=-1)
    
    
class Collector:
    def __init__(self):
        self.run_data = {}
        self.all_data = {}

    def reset(self):
        for k in self.run_data:
            # if there's already an array get that
            # otherwise construct a new empty array
            arr = self.all_data.get(k, [])
            arr.append(self.run_data[k])

            # put the array back in case we were working with a new array
            self.all_data[k] = arr

        # reset the run_data for the next run
        self.run_data = {}

    def collect(self, name, value):
        arr = self.run_data.get(name, [])
        arr.append(value)

        self.run_data[name] = arr

    def getStats(self, name):
        arr = self.all_data[name]

        runs = len(arr)

        arr = np.array(arr)
        mean = np.mean(arr, axis=0)
        stderr = np.std(arr, axis=0, ddof=1) / np.sqrt(runs)

        return [mean, stderr, runs]
    
    def getStats_list(self, name):
        arr = self.all_data[name]

        runs = len(arr)

        arr = np.array(arr)
        # mean = np.mean(arr, axis=0)
        # stderr = np.std(arr, axis=0, ddof=1) / np.sqrt(runs)
        
        mean, stderr = tolerant_mean(arr)

        return [mean, stderr, runs]
