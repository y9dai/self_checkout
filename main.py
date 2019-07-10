import argparse
import cv2
import numpy as np
import sys
import os
# ライブラリまでのディレクトリ定義
sys.path.append('./libs')
from init_camera import init_camera, get_image
from detect import detect
from classify import categorical_pred, binary_pred

total_price = 0
bottle_count = 0
bottle_str = 'bottle'
desc_text = 'Please place the bottle in the blue box.'
scan_text = 'scaning...'
push_text = 'Please press the Enter-key.'
img_path = 'tmp/data.jpg'

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="number of video device", default=0)
parser.add_argument("--prototxt", default="data/mobilenet_v2_deploy.prototxt")
parser.add_argument("--caffemodel", default="models/mobilenet_v2.caffemodel")
parser.add_argument("--classNames", default="data/synset.txt")
parser.add_argument("--preview", default=True)
parser.add_argument("--picamera", default=True)
parser.add_argument("--auto", default=False)


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
                key = input('続けて商品をスキャンする場合は「y + Enter」,会計する場合は「Enter」を押して下さい')

                if key != 'y':
                    print("合計:{}円".format(total_price))
                    total_price = 0

            os.remove(img_path)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
