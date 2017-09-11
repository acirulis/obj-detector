1. Setup python virtualenv -p python3
2. installed numpy, flask, opencv-python
3. turned on GPU support in Makefile
5. edited darknet/cfg/coco.data
6. LD_LIBRARY_PATH env variable both for CUDA and libdarknet.so
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
7. added /home/whitedigital/Andis/vision/darknet/__init__.py
8. wget https://pjreddie.com/media/files/yolo.weights
9.

IP camera: rtsp://admin:123456@192.168.0.76/11
IPC3612ER3-PF28-B
IPC3612ER3-PF28(40)(60)-B
 

./darknet detector demo data/voc.data yolo-voc.cfg yolo-voc.weights rtsp://admin:123456@192.168.0.76/11 -i 0
./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights rtsp://admin:123456@192.168.0.76/media/video2 -i 0

http://www.chioka.in/python-live-video-streaming-example/

http://blog.mycodesite.com/compile-opencv-with-ffmpeg-for-ubuntudebian/
