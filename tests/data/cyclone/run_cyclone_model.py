import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':

    input_data_ref_C = np.load("cyclone_input_200x200.npy")

    # save inputs in C and F style
    input_data_ref_F = np.asfortranarray(input_data_ref_C)
    np.savetxt("cyclone_input_200x200_ctensor.csv", input_data_ref_C.flatten(order="K"), delimiter="\n")
    np.savetxt("cyclone_input_200x200_ftensor.csv", input_data_ref_F.flatten(order="K"), delimiter="\n")

    # read data from CSV and run the model
    model = tf.saved_model.load("cyclone_model_200x200_tf")

    input_data = np.loadtxt("cyclone_input_200x200_ctensor.csv", dtype=np.float32)
    input_data = input_data.reshape((1, 200, 200, 17))

    prediction = model(input_data)

    pnp = np.asarray(prediction.numpy())
    pnp = pnp.reshape((200, 200))
    plt.imshow(pnp)
    plt.savefig("prediction_c.png")

    pnp_f = np.asfortranarray(pnp)
    pnp_f = pnp_f.flatten(order="K").reshape((200, 200))
    plt.imshow(pnp_f)
    plt.savefig("prediction_f.png")

    np.savetxt("cyclone_output_200x200_ctensor.csv", pnp.flatten(order="K"), delimiter="\n")
    np.savetxt("cyclone_output_200x200_ftensor.csv", pnp_f.flatten(order="K"), delimiter="\n")
