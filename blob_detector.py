import cv2


class BlobDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = False
        params.minArea = 3000
        params.maxArea = 6000
        params.filterByCircularity = True
        params.minCircularity = 0.5
        self.detector = cv2.SimpleBlobDetector_create(params)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(img_gray, 9, 25, 25)
            canny = cv2.Canny(filtered, 50, 200)
            points = self.detector.detect(canny)
            for point in points:
                x, y = int(point.pt[0]), int(point.pt[1])
                s = point.size
                r = int(s//2)
                cv2.circle(img, (x, y), r, (255, 255, 0), 2)
            cv2.imshow('Image', img)
            cv2.waitKey(100)


if __name__ == '__main__':
    test = BlobDetector()
    test.run()
