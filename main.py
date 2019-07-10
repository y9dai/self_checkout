import argparse
import cv2
from cv2 import dnn
import numpy as np
import sys
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image


total_price = 0
bottle_count = 0
in_scale_factor = 0.017
in_width = 224
in_height = 224
mean_val = (103.94, 116.78, 123.68)
bottle_str = 'bottle'
desc_text = 'Please place the bottle in the blue box.'
scan_text = 'scaning...'
push_text = 'Please press the Enter-key.'
img_path = 'tmp/data.jpg'
categorical_model = load_model('models/MobileNetV2_shape224.h5')

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="number of video device", default=0)
parser.add_argument("--prototxt", default="data/mobilenet_v2_deploy.prototxt")
parser.add_argument("--caffemodel", default="models/mobilenet_v2.caffemodel")
parser.add_argument("--classNames", default="data/synset.txt")
parser.add_argument("--preview", default=True)
parser.add_argument("--picamera", default=True)
parser.add_argument("--auto", default=False)


def init_camera(picam_flg = True):
    cap = None
    res = False
    while res is False:
        if picam_flg:
            import picamera

            cap = picamera.PiCamera()
            # cap.start_preview()
            cap.resolution = (640, 480)
            cap.framerate = 33
            cv2.waitKey(1000)
            res = True
        else:
            cap = cv2.VideoCapture(0)
            res, _ = cap.read()
            #pass
            cv2.waitKey(1000)
            print('retry ..')
    return cap


def get_image(cap, picam_flg = True):
    c_frame = None
    if picam_flg:
        import picamera.array

        with picamera.array.PiRGBArray(cap, size=(640, 480)) as stream:
            c_frame = cap.capture(stream, 'bgr')
            c_frame = stream.array
            if c_frame is None:
                exit

    else:
        end_flag, c_frame = cap.read()
        if end_flag is False or c_frame is None:
            exit
    return c_frame


def detect(rgb_frame, class_names, prototxt, caffemodel):
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    blob = dnn.blobFromImage(rgb_frame, in_scale_factor, (in_width, in_height), mean_val)
    net.setInput(blob)
    detections = net.forward()

    max_class_id = 0
    max_class_point = 0;
    for i in range(detections.shape[1]):
        class_point = detections[0, i, 0, 0]
        if (class_point > max_class_point):
            max_class_id = i
            max_class_point = class_point

    return class_names[max_class_id]


def categorical_pred():
    label = ['ayataka', 'cocacola', 'craft_boss_black', 'energy_peaker', 'irohas']
    money = [110, 120, 130, 140, 150]

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


def binary_pred(drink_name):
    result = False
    # モデル+重みを読込み
    binary_model = load_model('models/' + drink_name + '.h5')


    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array.astype('float32')/255.0
    img_array = img_array.reshape((1,224,224,3))

    # predict
    img_pred = np.argmax(binary_model.predict(img_array))
    print(img_pred)
    if img_pred == 0:
      result = True

    return result


if __name__ == "__main__":
    args = parser.parse_args()
    picam_flg = (args.picamera == True or args.picamera == "True" or args.picamera == "true")
    auto_flg = (args.auto == True or args.auto == "True" or args.auto == "true")
    showPreview = (args.preview == True or args.preview == "True" or args.preview == "true")
    f = open(args.classNames, 'r')
    class_names = f.readlines()
    cap = init_camera(picam_flg)

    while True:
        frame = get_image(cap, picam_flg)
        dst = frame[55:695, 545:795]
        cv2.rectangle(frame, (550, 50), (800, 700), (255, 0, 0))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        class_name = detect(rgb_frame, class_names, args.prototxt, args.caffemodel)

        if bottle_str in class_name:
            if auto_flg:
                bottle_count += 1
                cv2.putText(frame, scan_text, (10, 30), font, size, color, weight)
                cv2.imshow("detections", frame)
                if bottle_count >= 30:
                    bottle_count = 0
                    cv2.imwrite(img_path, dst)
            else:
                cv2.putText(frame, push_text, (10, 30), font, size, color, weight)
                cv2.imshow("detections", frame)
                if cv2.waitKey(1) & 0xFF == 13:
                    cv2.imwrite(img_path, dst)

        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            size = 1
            color = (255,255,255)
            weight = 2
            cv2.putText(frame, desc_text, (10, 30), font, size, color, weight)
            cv2.imshow("detections", frame)

        if os.path.exists(img_path):
            drink_name, drink_price = categorical_pred()

            if binary_pred(drink_name):
                print('{} : {}RWF'.format(drink_name, drink_price))
                total_price += drink_price
                key = input('Press "y + Enter" to scan products continuously, or "Enter" to check')

                if key != 'y':
                    print("合計:{}円".format(total_price))
                    os.remove(img_path)
                    break

            os.remove(img_path)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
