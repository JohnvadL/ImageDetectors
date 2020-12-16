import os
import sys

import dlib
import numpy as np
from PIL import Image

'''
Takes in an image and returns 
'''


class eyes_closed_detector(object):

    def __call__(self, image):

        image = np.asarray(image)

        predictor_path = os.path.join(sys.path[0], "../shape_predictor_68_face_landmarks.dat")
        shape_predictor = dlib.shape_predictor(predictor_path)
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(image)

        if not faces:
            raise ValueError("Image has no detectable faces")

        # assumption is made that there is only one
        for face in faces:
            landmarks = shape_predictor(image, face)

            landmarks = landmarks.parts()
            landmarks = self.convert_to_np(landmarks)

            # eye landmarks
            left = landmarks[36:42]
            right = landmarks[42:48]

            EAR_left = self.EAR(left)
            EAR_right = self.EAR(right)

            EAR_left = EAR_left < 0.19
            EAR_right = EAR_right < 0.19

            return EAR_left, EAR_right


    @staticmethod
    def convert_to_np(points):
        np_points = np.array([], dtype=np.int)
        while points:
            point = points.pop()
            np_points = np.append(np_points, (point.x, point.y))

        np_points = np_points.reshape((-1, 2))
        np_points = np.flip(np_points, axis=0)
        return np_points

    @staticmethod
    def EAR(eyes):
        a = np.linalg.norm(eyes[1] - eyes[5], ord=2)
        b = np.linalg.norm(eyes[2] - eyes[4], ord=2)
        c = np.linalg.norm(eyes[0] - eyes[3], ord=2)
        return (a + b) / (2 * c)
