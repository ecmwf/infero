import os
import numpy as np
import argparse
import keras
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":
    """
    Lightweight script to convert a keras model into a TFlite model
    """
    
    parser = argparse.ArgumentParser("Data Augmentation")
    
    parser.add_argument('keras_model_path', help="Path of the input keras model")
    
    parser.add_argument('tflite_model_path', help="Path of the output tflite model")
    
    parser.add_argument("--verify_with", help="Check the model by passing an input numpy path")
    
    args = parser.parse_args()
    
    # Load the model
    keras_model = keras.models.load_model(args.keras_model_path)
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model_path = converter.convert()
    
    # Save the model.
    with open(args.tflite_model_path, 'wb') as f:
        f.write(tflite_model_path)
    
    # verify the model if required
    if args.verify_with is not None:
        input_data = np.load(args.verify_with)
        pred = keras_model(input_data[0, :].reshape((1, -1)))
        print(f"pred {pred}")


    
    

