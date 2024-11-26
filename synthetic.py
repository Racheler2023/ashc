import random
import numpy as np

# Randomly generate data set method
def random_dataset(n=20, d=3, bound=8):
    random_seed = random.randint(0, 2**32 - 1)
    np.random.seed(random_seed)
    unique_dataset = set()
    while len(unique_dataset) < n:
        # Generate a random number between 0 and bound
        new_points = np.random.uniform(0, bound, (n, d))
        for point in new_points:
            if len(unique_dataset) < n:
                unique_dataset.add(tuple(point))
            else:
                break
    return np.array(list(unique_dataset))
