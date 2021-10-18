from tracking import FaceTracking

import cv2
import threading
import queue


class Camera:

    def __init__(self, idx=None):

        self.idx = int(idx)

        ## Camera thread
        self.thread_cam = threading.Thread(target=self.camera)

        ## Detection thread
        self.detection_on = False
        self.thread_det = threading.Thread(target=self.faceTracking)

        ## Camera frames queue
        self.camera_frames = queue.Queue()
        self.camera_frames.put(1)

        ## Faces coord
        self.faces_coord = queue.Queue()

        ## Face tracking method
        self.tr = FaceTracking(self.faces_coord)
        self.track = []

    def start_camera(self):
        self.thread_cam = threading.Thread(target=self.camera)
        self.thread_cam.start()

    def start_detection(self):
        self.detection_on = not self.detection_on
        if self.detection_on == True:
            print("--- Starting Tracking ---")
            self.thread_det = threading.Thread(target=self.faceTracking)
            self.thread_det.start()

    def camera(self):

        print("--- Opening Camera ---")

        fn = self.idx
        cam = cv2.VideoCapture(fn)
        detection = False

        if cam is None or not cam.isOpened():
            print('Can´t open camera:', self.idx)

        else:
            while True:

                ret, img = cam.read()

                if ret == True:

                    img = cv2.resize(img, (640, 480))
                    self.camera_frames.put(img)

                    ## draw faces
                    if len(self.track) > 0:
                        for i, t in enumerate(self.track):
                            x, y, w, h = t['coord']
                            if t['detected']:
                                color = (0, 255, 0)
                            else:
                                color = (0, 0, 255)

                            dh = int(h/3)
                            dw = int (w/3)

                            ## draw horizontal lines
                            cv2.line(img, (x, y), (x + dw, y), color, 3)
                            cv2.line(img, (x + 2 * dw, y), (x + w, y), color, 3)
                            cv2.line(img, (x, y + h), (x + dw, y + h), color, 3)
                            cv2.line(img, (x + 2 * dw, y + h), (x + w, y + h), color, 3)

                            ## draw vertical lines
                            cv2.line(img, (x, y), (x, y + dh), color, 3)
                            cv2.line(img, (x, y + 2 * dh), (x , y + h), color, 3)
                            cv2.line(img, (x + w, y), (x + w, y + dh), color, 3)
                            cv2.line(img, (x + w, y + 2 * dh), (x + w, y + h), color, 3)

                    if detection:
                        cv2.putText(img, 'Tracking ON', (1, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(img, 'Tracking OFF (press d key)', (1, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                    cv2.imshow('Camera', img)
                    key = cv2.waitKey(1) & 0xFF

                    ## close camera
                    if key == ord("q"):
                        break

                    ## start/stop detection
                    if key == ord("d"):
                        detection = not detection
                        self.start_detection()
                else:
                    cam = cv2.VideoCapture(fn)

            ## close windows
            cam.release()
            cv2.destroyAllWindows()

            ## clean camera frames queue
            self.camera_frames.queue.clear()
            self.camera_frames.put(0)

        print("--- Camera Closed ---")

    def faceTracking(self):

        while True:

            ## stop detection
            if not self.detection_on:
                self.track = []
                break

            ## get camera frames
            img = self.camera_frames.get()

            ## ignore null frames
            if type(img) == int:
                if img != 1:
                    break
            else:

                if img.shape[2] == 3:
                    ## tracking face coordinates
                    self.track = self.tr.detectar_caras(img)
                    ## clean camera frames queue
                    self.camera_frames.queue.clear()

        print("--- Tracking Finished ---")



if __name__ == "__main__":
    det = Camera(idx=0)
    det.start_camera()
