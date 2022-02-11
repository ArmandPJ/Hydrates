import numpy as np

arr = np.array([2, 3, 1])
print(arr)

print(f"Arctangent: {np.arctan(arr[1]/arr[0])}")
print(f"Arccosine:  {np.arccos(arr[2]/np.linalg.norm(arr))}")