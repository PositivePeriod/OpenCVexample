import cv2


class ColorHistogramFilter:
    def back_projection(self, img, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_total = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        roi_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        dst = cv2.calcBackProject([hsv_total], [0, 1], roi_hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thr = cv2.threshold(dst, 50, 255, 0)
        thr = cv2.merge((thr, thr, thr))
        result = cv2.bitwise_and(img, thr)
        return result

    def main(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Original')
        cv2.namedWindow('Filtered')
        while True:
            ret, img1 = cap.read()
            if ret:
                cv2.imshow('Original', img1)

                while True:
                    roi = cv2.selectROI('Original', img1)
                    print('roi', roi)
                    img2 = self.back_projection(img1, roi)
                    cv2.imshow('Filtered', img2)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test = ColorHistogramFilter()
    test.main()
