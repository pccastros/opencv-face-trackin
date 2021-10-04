import cv2
import threading
import queue
import math
import numpy as np

class tracking:

    def __init__(self, faces_coord):

        self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.track = []
        self.min_frames_new_face = 6
        self.max_frames_disappear = 6
        self.min_pixel_width_face = 150

        self.faces_coord = faces_coord

    def distancia(self, pt1, pt2):

        return math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))

    def faceData(self, face_center):

        data = {
            'center': face_center,   ## center of face
            'status': 0,   ## face status
            'detection': 0,     ## face detection status
            'aux_counter': 0,     ## counter
            'coord': [0, 0, 0, 0],     ## face coordinates
            'detected': False    ## face detected
        }
        return data

    def detectar_caras(self, image):

        ## opencv haar cascades detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        ## if face detected
        if len(faces) > 0:

            face = np.array([0, 0, 0, 0])
            for i, f in enumerate(faces):
                (x, y, w, h) = f
                if (w * h) > face[2] * face[3]:
                    face = f

            (x, y, w, h) = face

            ## coordinates of center
            face_center = (int(x + w / 2), int(y + h / 2))

            ## if previous faces
            if len(self.track) > 0:

                ## looks for previus detections
                for j, tr in enumerate(self.track):

                    ## get status and center coord
                    st = tr['status']
                    ct = tr['center']

                    if st == 2 or st == 1:

                        ## calcule distance between previus and actual center coords
                        d = self.distancia(ct, face_center)

                        ## if distance less than
                        if d < max(w, h):

                            ## if face was detected
                            if st == 2:
                                tr['center'] = face_center
                                tr['status'] = 3
                                tr['aux_counter'] = 0
                                tr['coord'] = [x, y, w, h]

                                break

                            ## if face in validating period
                            else:
                                self.track[j]['center'] = face_center
                                self.track[j]['status'] = 0
                                self.track[j]['coord'] = [x, y, w, h]
                                break

                    ## new face detected
                    elif j == len(self.track) - 1:

                        face_data = self.faceData(face_center)
                        self.track.append(face_data)
                        break

            ## new face detected
            else:

                face_data = self.faceData(face_center)
                self.track.append(face_data)

        if len(self.track) > 0:

            """
            FACE STATUS
                0: new face appear
                1: face in detection validation
                2: face detected


                2: face detected and validate time in camera
                3: face detected and validate time in camera
                4: delete face data 
            """

            """
            FACE DETECTION
                0: no face detected
                2: face detected
                
                1: 
                2: detection True
                3: first time detected

            """

            for i, face in enumerate(self.track):

                status = face['status']

                ## new face detected
                if status == 0:

                    ## face steel in frame
                    if self.track[i]['detection'] == 0:
                        self.track[i]['status'] = 1
                        self.track[i]['aux_counter'] += 1

                        ## if face is in frame more than time thresh
                        if self.track[i]['aux_counter'] > self.min_frames_new_face:
                            self.track[i]['status'] = 2
                            self.track[i]['aux_counter'] = 0

                    ## face detection True
                    elif self.track[i]['detection'] == 2:
                        self.track[i]['aux_counter'] = 0

                    self.track[i]['detection'] = 0


                ## face in validating period
                elif status == 1:

                    ## if face was detected and disappear
                    if self.track[i]['detection'] == 2:
                        self.track[i]['aux_counter'] += 1

                        if self.track[i]['aux_counter'] > self.max_frames_disappear:
                            self.track[i]['status'] = 4

                    ## if face in validating period and disappear
                    elif self.track[i]['detection'] == 0:
                        self.track[i]['aux_counter'] = 0

                    self.track[i]['detection'] = 2

                ## face detection complete
                elif status == 2:
                    self.track[i]['aux_counter'] += 1

                    if self.track[i]['aux_counter'] > self.max_frames_disappear:
                        self.track[i]['status'] = 4

                    self.track[i]['detection'] = 1

                ## face detection complete
                elif status == 3:
                    self.track[i]['status'] = 2
                    self.track[i]['detection'] = 3

                    if self.track[j]['detected'] == False:
                        tx, ty, tw, th = self.track[j]['coord']
                        if tw > self.min_pixel_width_face:
                            self.track[j]['detected'] = True
                            self.faces_coord.put(("Persona Detectada " + str(tw)))

                ## face detection complete
                elif status == 4:
                    if len(self.track) > 0:
                        del self.track[i]


        return self.track


class detection:

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
        self.tr = tracking(self.faces_coord)
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

det = detection(idx=0)
det.start_camera()
