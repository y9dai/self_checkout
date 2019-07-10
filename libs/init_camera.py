import cv2

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
