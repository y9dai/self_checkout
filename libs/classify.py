from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

img_path = 'tmp/data.jpg'


def categorical_pred(categorical_model):
    label = ['ayataka', 'cocacola', 'craft_boss_black', 'energy_peaker', 'ilohas', 'unknown']
    money = [1680, 1250, 1800, 2080, 1460, 0]
    threshold = 0.8 #threshold以下のものは全てunknownにする

    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array.astype('float32')/255.0
    img_array = img_array.reshape((1,224,224,3))

    # predict
    img_proba = categorical_model.predict(img_array)
    print(img_proba)
    drink_index = np.argmax(img_proba)

    #unknownだったらそのまま返す
    if (drink_index == 5):
        pass

    #unknown以外だったら、閾値で判定してunknownにするか、そのまま返すかする
    else:
        if (img_proba[0][drink_index] < threshold):
            drink_index = 5 #unknown(5)に設定
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


