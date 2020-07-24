import cv2
import numpy as np

# https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
# https://opencv-python.readthedocs.io/en/latest/doc/05.trackBar/trackBar.html


class HSVFilterSimulator:
    def __init__(self):
        self.image = None
        self.palette = None

        # Camera setting
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Create a window
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse)
        cv2.namedWindow('option')
        cv2.namedWindow('palette')

        # Create track bars for color change
        # Hue is from 0-179 for OpenCV
        cv2.createTrackbar('HMin', 'option', 0, 179, self.nothing)
        cv2.createTrackbar('SMin', 'option', 0, 255, self.nothing)
        cv2.createTrackbar('VMin', 'option', 0, 255, self.nothing)
        cv2.createTrackbar('HMax', 'option', 0, 179, self.nothing)
        cv2.createTrackbar('SMax', 'option', 0, 255, self.nothing)
        cv2.createTrackbar('VMax', 'option', 0, 255, self.nothing)

        # Set default value for Max HSV track bars
        cv2.setTrackbarPos('HMax', 'option', 179)
        cv2.setTrackbarPos('SMax', 'option', 255)
        cv2.setTrackbarPos('VMax', 'option', 255)

        # Initialize HSV min/max values
        self.range = {'HMin': 0, 'SMin': 0, 'VMin': 0, 'HMax': 0, 'SMax': 0, 'VMax': 0}
        self.range_ = {'HMin': 0, 'SMin': 0, 'VMin': 0, 'HMax': 0, 'SMax': 0, 'VMax': 0}

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            print(f'HSV of the clicked pixel; {img[y][x]}')
            self.palette = self.create_blank(img[y][x])
            cv2.imshow('palette', self.palette)

    def create_blank(self, hsv, width=200, height=200):
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)

        # Fill image with color
        image[:] = hsv

        # Change hsv to bgr
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def nothing(self, event):
        pass

    def main(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                # Flip left and right
                self.image = cv2.flip(frame, 1)

                # Get current positions of all track bars
                for x in self.range.keys():
                    self.range[x] = cv2.getTrackbarPos(x, 'option')

                # Set minimum and maximum HSV values to display
                lower = np.array([self.range['HMin'], self.range['SMin'], self.range['VMin']])
                upper = np.array([self.range['HMax'], self.range['SMax'], self.range['VMax']])

                # Convert to HSV format and color threshold
                hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_image, lower, upper)
                self.image = cv2.bitwise_and(self.image, self.image, mask=mask)

                # Print if there is a change in HSV value
                if self.range != self.range_:
                    print(f"Min : {self.range['HMin']} {self.range['SMin']} {self.range['VMin']}")
                    print(f"Max : {self.range['HMax']} {self.range['SMax']} {self.range['VMax']}")
                    print()
                    self.range_ = self.range.copy()

                # Display result image
                cv2.imshow('image', self.image)

                # Exit if q is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break


if __name__ == '__main__':
    test = HSVFilterSimulator()
    test.main()
