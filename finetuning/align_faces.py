import os

import cv2
import dlib
import numpy as np
from PIL import Image


class align_faces(object):

    def __call__(self, image):

        # Since the images are loaded it PIL
        image = np.asarray(image)

        # The percentage value of how far in the picture the left eye should be
        LEFT_EYE_CORD = (0.25, 0.2)
        DIMENSIONS = 244

        train_folder = os.getcwd()
        predictor_path = os.path.join(train_folder, "shape_predictor_5_face_landmarks.dat")
        # detector_path = os.path.join(train_folder, "mmod_human_face_detector.dat")
        shape_predictor = dlib.shape_predictor(predictor_path)
        face_detector = dlib.get_frontal_face_detector()

        # changed here for efficiency
        # face_detector = dlib.cnn_face_detection_model_v1(detector_path)
        faces = face_detector(image)

        if not faces:
            raise ValueError("Image has no detectable faces")

        # assumption is made that there is only one
        for face in faces:
            landmarks = shape_predictor(image, face)
            landmarks = landmarks.parts()
            landmarks = self.convert_to_np(landmarks)

            # To Gauge Scale
            maximum = np.max(landmarks, axis=0)
            minimum = np.min(landmarks, axis=0)

            # eye landmarks
            right = landmarks[:2]
            left = landmarks[2:4]

            left = np.mean(left, axis=0, dtype=np.int)
            right = np.mean(right, axis=0, dtype=np.int)

            centre = np.vstack((left, right))
            centre = np.mean(centre, axis=0, dtype=np.int)

            diff = right - left
            diff = diff.reshape((2, 1))

            angle = np.degrees(np.arctan2(diff[1], diff[0]))

            # find the length of the face, and use that for our scale
            y_scale = maximum[1] - minimum[1]
            y_scale = y_scale + 2.1 * y_scale

            M = cv2.getRotationMatrix2D((centre[0], centre[1]), angle, DIMENSIONS / y_scale)

            # update translation
            t_x = DIMENSIONS // 2
            t_y = DIMENSIONS * LEFT_EYE_CORD[1]
            M[0, 2] += (t_x - centre[0])
            M[1, 2] += (t_y - centre[1])

            image2 = cv2.warpAffine(image, M, (DIMENSIONS, DIMENSIONS),
                                    flags=cv2.INTER_CUBIC)

            # convert back to PIL
            return Image.fromarray(image2)

    @staticmethod
    def convert_to_np(points):
        np_points = np.array([], dtype=np.int)
        while points:
            point = points.pop()
            np_points = np.append(np_points, (point.x, point.y))

        np_points = np_points.reshape((-1, 2))
        np_points = np.flip(np_points, axis=0)
        return np_points
