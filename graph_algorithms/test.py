import numpy as np

n = 1000
print(np.random.random(n))
vals = np.random.random(n).reshape(-1, 1) 
print(vals)

ratings_dist = np.array([0.05, 0.05, .05, .45, .4])
print(vals < ratings_dist.cumsum())
ratings = 1 + np.argmax(vals < ratings_dist.cumsum(), axis=1)
print(ratings)
print(ratings_dist.cumsum())
print(ratings.mean())
