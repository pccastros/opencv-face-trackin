import cv2
import math
import math
import numpy as np


class FaceTracking:

    def __init__(self, faces_coord):

        self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.faces = []
        self.min_frames_new_face = 6
        self.max_frames_disappear = 6
        self.min_pixel_width_face = 150
        self.faces_coord = faces_coord

    def face_distance(self, pt1, pt2):
        return math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))


    def face_data(self, face_center):

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
            if len(self.faces) > 0:

                ## looks for previus detections
                for j, tr in enumerate(self.faces):

                    ## get status and center coord
                    st = tr['status']
                    ct = tr['center']

                    if st == 2 or st == 1:

                        ## calcule distance between previus and actual center coords
                        d = self.face_distance(ct, face_center)

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
                                self.faces[j]['center'] = face_center
                                self.faces[j]['status'] = 0
                                self.faces[j]['coord'] = [x, y, w, h]
                                break

                    ## new face detected
                    elif j == len(self.faces) - 1:

                        face_data = self.face_data(face_center)
                        self.faces.append(face_data)
                        break

            ## new face detected
            else:

                face_data = self.face_data(face_center)
                self.faces.append(face_data)

        if len(self.faces) > 0:

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

            for i, face in enumerate(self.faces):

                status = face['status']

                ## new face detected
                if status == 0:

                    ## face steel in frame
                    if self.faces[i]['detection'] == 0:
                        self.faces[i]['status'] = 1
                        self.faces[i]['aux_counter'] += 1

                        ## if face is in frame more than time thresh
                        if self.faces[i]['aux_counter'] > self.min_frames_new_face:
                            self.faces[i]['status'] = 2
                            self.faces[i]['aux_counter'] = 0

                    ## face detection True
                    elif self.faces[i]['detection'] == 2:
                        self.faces[i]['aux_counter'] = 0

                    self.faces[i]['detection'] = 0


                ## face in validating period
                elif status == 1:

                    ## if face was detected and disappear
                    if self.faces[i]['detection'] == 2:
                        self.faces[i]['aux_counter'] += 1

                        if self.faces[i]['aux_counter'] > self.max_frames_disappear:
                            self.faces[i]['status'] = 4

                    ## if face in validating period and disappear
                    elif self.faces[i]['detection'] == 0:
                        self.faces[i]['aux_counter'] = 0

                    self.faces[i]['detection'] = 2

                ## face detection complete
                elif status == 2:
                    self.faces[i]['aux_counter'] += 1

                    if self.faces[i]['aux_counter'] > self.max_frames_disappear:
                        self.faces[i]['status'] = 4

                    self.faces[i]['detection'] = 1

                ## face detection complete
                elif status == 3:
                    self.faces[i]['status'] = 2
                    self.faces[i]['detection'] = 3

                    if self.faces[j]['detected'] == False:
                        tx, ty, tw, th = self.faces[j]['coord']
                        if tw > self.min_pixel_width_face:
                            self.faces[j]['detected'] = True
                            self.faces_coord.put(("Face detected " + str(tw)))

                ## face detection complete
                elif status == 4:
                    if len(self.faces) > 0:
                        del self.faces[i]


        return self.faces