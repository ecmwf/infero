import os
import numpy as np
import pyinfero

# config
this_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(this_dir, "../../../../../tests/data/cyclone")
input_path = os.path.join(data_dir, "cyclone_input_200x200.npy")
model_path = os.path.join(data_dir, "cyclone_model_200x200.tflite") 
model_type = "tflite" 
model_output_shape = (1,200,200,1) 

# load input
input_tensor = np.load(input_path)

# inference
infero = pyinfero.Infero(model_path, model_type)
infero.initialise()
output_tensor = infero.infer(input_tensor, model_output_shape)
infero.finalise()

# save output
np.save("output.npy", output_tensor)

print("all done.")

