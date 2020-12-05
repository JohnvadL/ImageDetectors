import cv2
import numpy as np
from PIL import Image

'''
References:
Preprocessing steps: https://arxiv.org/pdf/1911.05946.pdf
Face landmarks : https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/C
'''


class align_faces_with_landmarks(object):

    def __call__(self, image, landmarks):
        # Since the images are loaded it PIL
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # The percentage value of how far in the picture the left eye should be
        LEFT_EYE_CORD = (0.25, 0.2)
        DIMENSIONS = 244

        landmarks = np.array(landmarks).reshape((5, 2))
        # assumption is made that there is only one

        # To Gauge Scale
        maximum = np.max(landmarks, axis=0)
        minimum = np.min(landmarks, axis=0)

        # eye landmarks
        left = landmarks[:1]
        right = landmarks[1:2]

        centre = np.vstack((left, right))
        centre = np.mean(centre, axis=0, dtype=np.int)

        diff = right - left
        diff = diff.reshape((2, 1))

        angle = np.degrees(np.arctan2(diff[1], diff[0]))

        # find the length of the face, and use that for our scale
        y_scale = maximum[1] - minimum[1]
        y_scale = y_scale + 0.9 * y_scale

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


