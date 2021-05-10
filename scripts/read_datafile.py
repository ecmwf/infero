import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    assert len(sys.argv) == 3
except AssertionError:
    raise AssertionError(f" {len(sys.argv)} python read_data <filename> <channel>")

data_file = sys.argv[1]
ich = int(sys.argv[2])

print(f"Selected: data_file {data_file}, channel {ich}")

vv = np.load(data_file)
print(f"shape {vv.shape}")
print(f"max {np.max(vv)}")
print(f"min {np.min(vv)}")
print(f"mean {np.mean(vv)}")

print(f"\n-----------------")
print(f"press q to exit..")
print(f"-----------------")

plt.imshow(vv[0, :, :, ich])
plt.show()

