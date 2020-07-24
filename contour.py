import numpy as np
import cv2

# https://bretahajek.com/2017/01/scanning-documents-photos-opencv/


class Contour:
    def make_hsv_mask(self, image, lower, upper):
        img_adapted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_masked = cv2.inRange(img_adapted, lower, upper)  # filter by skin color
        img_blurred = cv2.blur(img_masked, (7, 7))
        _, img_thr = cv2.threshold(img_blurred, 127, 255, cv2.THRESH_BINARY_INV)  # remove noise
        return img_thr

    def get_contour(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        biggest_contour = sorted_contours[0]
        return biggest_contour


if __name__ == '__main__':
    test = Contour()
    cap = cv2.VideoCapture(0)
    # experimental skin range - HSV
    skin_color_upper = np.array([179, 79, 255])
    skin_color_lower = np.array([0, 0, 104])
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img_mask = test.make_hsv_mask(frame, skin_color_lower, skin_color_upper)
            contour = test.get_contour(img_mask)

            # draw contour to original image
            cv2.drawContours(frame, contour, -1, (0, 0, 255), 2)
            cv2.imshow('frame', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
