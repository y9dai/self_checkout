from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

img_path = 'tmp/data.jpg'


def categorical_pred(categorical_model):
    label = ['ayataka', 'cocacola', 'craft_boss_black', 'energy_peaker', 'ilohas', 'unknown']
    money = [1680, 1250, 1800, 2080, 1460, 0]

    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array.astype('float32')/255.0
    img_array = img_array.reshape((1,224,224,3))

    # predict
    img_pred = categorical_model.predict(img_array)
    drink_index = np.argmax(img_pred)
    drink_name = label[drink_index]
    drink_price = money[drink_index]

    return drink_name, drink_price


def binary_pred(binary_model):
    result = True

    if not binary_model is None:
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = img_array.astype('float32')/255.0
        img_array = img_array.reshape((1,224,224,3))

        # predict
        img_pred = np.argmax(binary_model.predict(img_array))

        if img_pred != 0:
          result = False

    return result


