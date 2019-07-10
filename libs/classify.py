from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

img_path = 'tmp/data.jpg'

def categorical_pred():
    # モデル+重みを読込み
    self_model = load_model('models/MobileNetV2_shape224.h5')

    label = ['ayataka', 'cocacola', 'craft_boss_black', 'energy_peaker', 'irohas']
    money = [110, 120, 130, 140, 150]

    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array.astype('float32')/255.0
    img_array = img_array.reshape((1,224,224,3))

    # predict
    img_pred = self_model.predict(img_array)
    drink_index = np.argmax(img_pred)
    drink_name = label[drink_index]
    drink_price = money[drink_index]

    return drink_name, drink_price

def binary_pred(drink_name):
    result = False
    # モデル+重みを読込み
    self_model = load_model(drink_name + '.h5')


    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array.astype('float32')/255.0
    img_array = img_array.reshape((1,224,224,3))

    # predict
    img_pred = self_model.predict(img_array)

    if img_pred == 0:
      result = True

    return result


