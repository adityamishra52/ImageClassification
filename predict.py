import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

try:
    print("-"*100)
    model_path='models/my_first_model.keras'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found . check your train.py first?")
        exit()
    model=tf.keras.models.load_model(model_path)

    # define the categories
    class_name=['boy','cat','dog','girl','man']



except Exception as e:
    print("Error in initilizes model: ",e)

# create function for prediction
try:
    def predict_image(img_path):
        img=image.load_img(img_path,target_size=(224,224))

        img_array= image.img_to_array(img)
        img_array= tf.expand_dims(img_array,0)
        prediction=model.predict(img_array)
        score=tf.nn.softmax(prediction[0])

        print("\nThis image most likely belongs to {} with a {:.2f} percent confidence."
              .format(class_name[np.argmax(score)], 100 * np.max(score)))
        
except Exception as e:
    print("Error in defing prediction function: ",e)

test_image_path = 'data/test/test_photo.jpg' 

if os.path.exists(test_image_path):
    predict_image(test_image_path)
else:
    print(f"\nPlace a photo named '{test_image_path}' in your folder to test!")