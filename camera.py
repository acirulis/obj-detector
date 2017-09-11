import cv2


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture('rtsp://admin:123456@192.168.0.76/media/video2')
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def srink(img, factor=0.5):
        return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    def get_frame(self):
        success, image = self.video.read()
        # image = self.srink(image)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


if __name__ == '__main__':
    video = cv2.VideoCapture('rtsp://admin:123456@192.168.0.76/media/video2')
    import time
    time.sleep(10)
    ret, frame = video.read()

    cv2.imwrite('out2.jpeg', frame)

    video.release()
