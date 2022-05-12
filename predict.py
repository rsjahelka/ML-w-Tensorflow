from __future__ import absolute_import, division, print_function, unicode_literals
import json
import numpy as np
from PIL import Image
import argparse

model_path = './best_model.h5'
label_map = 'label_map.json'
image_path = '/test_images/'

def class_names(json_file):
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
        corrected_class_names = dict()
    for key in class_names:
        corrected_class_names[str(int(key)-1)] = class_names[key]
        return corrected_class_names

#load model
def load_model(model_path):
    keras_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})
    return keras_model
    
# process_image function
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = (image/255)
    image = image.numpy()
    return image

#predict function
def predict(image_path, model, top_k=5):
    image = np.asarray(im)
    processed_image = np.expand_dims(process_image(image), axis=0) 
    predictions = model.predict(processed_image)
    top_k_values, top_k_indices = tf.nn.top_k(predictions, k=top_k)
    top_k_indices = 1 + top_k_indices[0]    
    return top_k_values.numpy()[0], top_k_indices.numpy().astype(str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument("--image", help= "./test_images/", required=False, default="./test_images/wild_pansy.jpg")
    parser.add_argument("--model_path", help="best_model.h5", required=False, default="best_model.h5")
    parser.add_argument("--top_k", help="top k probs of the image", required=False, default=5)
    parser.add_argument("--category_names", help="classes", required=False, default=".label_map.json")
    args = vars(parser.parse_args())
    image_path = args['image']
    my_model = args['model_path']
    top_k = args['top_k']
    category_names = args['category_names']
    image_size = 224
    
    #python predict.py --image test_images/cautleya_spicata.jpg --model_path best_model.h5 --top_k 5 --category_names label_map.json