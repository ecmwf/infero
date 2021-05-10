import os
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
    
    parser.add_argument('--type',
                        help="Type of the input keras model",
                        choices=["native", "h5"],
                        default="native")
    
    parser.add_argument('tflite_model_path', help="Path of the output tflite model")

    args = parser.parse_args()
    
    keras_model = None
    temp_model_path="keras_model_temp"
    
    if args.type == "h5":
        keras_model = keras.models.load_model(args.keras_model_path)
        keras_model.save(temp_model_path)
    else:
        keras_model = args.keras_model_path

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(temp_model_path)
    tflite_model_path = converter.convert()

    # Save the model.
    with open(args.tflite_model_path, 'wb') as f:
        f.write(tflite_model_path)


    
    

