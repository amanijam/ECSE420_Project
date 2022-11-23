import ray

ray.init()

# Define the square task.
@ray.remote
def changep(p):
    p = [1, 2, 3] 

normal_p = []
ray.put(normal_p)
changep.remote(normal_p)
changed_p = ray.get(normal_p)

# Retrieve results.
print(normal_p)
print(changed_p)