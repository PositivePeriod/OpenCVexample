import numpy as np
import cv2


class Camshift:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def set_roi(self):
        cv2.namedWindow('Select ROI')
        while True:
            ret, image = self.cap.read()
            if ret:
                cv2.imshow('Select ROI', image)
                if cv2.waitKey(10) & 0xFF == ord(' '):
                    rect = cv2.selectROI('Select ROI', image, fromCenter=False, showCrosshair=True)
                    break

        # Enter space bar after selecting desired ROI
        cv2.destroyWindow('Select ROI')
        return rect

    def run(self, roi):
        ret, frame = self.cap.read()
        print(roi)
        x, y, w, h = roi
        track_window = (x, y, w, h)

        # set up the ROI for tracking
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        while True:
            ret, frame = self.cap.read()
            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                # apply meanshift to get the new location
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)

                # Draw it on image
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv2.polylines(frame, [pts], True, 255, 2)
                cv2.imshow('img2', img2)

                if cv2.waitKey(10) & 0xff == 27:
                    break
            else:
                break
        cv2.destroyAllWindows()
        self.cap.release()


if __name__ == '__main__':
    test = Camshift()
    roi_ = test.set_roi()
    test.run(roi_)