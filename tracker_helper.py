import cv2
import numpy as np

# https://wikidocs.net/48925


class TrackerHelper:
    def __init__(self, tracker, fit='height'):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.fit_to = fit

        # initialize tracker
        self.tracker_list = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create}

        self.tracker = self.tracker_list[tracker]()

    def set_roi(self):
        # main
        ret, image = self.cap.read()

        cv2.namedWindow('Select ROI')
        cv2.imshow('Select ROI', image)

        # Enter space bar after selecting desired ROI
        rect = cv2.selectROI('Select ROI', image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('Select ROI')
        return image, rect

    def run(self, init_image, init_roi, scale=1.5, size=(400, 400)):
        # global variables
        top_bottom_list, left_right_list = [], []

        # initialize tracker
        self.tracker.init(init_image, init_roi)

        while True:
            # read frame from video
            ret, image = self.cap.read()

            if not ret:
                print('No frame')
                break

            # update tracker and get position from new frame
            success, box = self.tracker.update(image)
            if not success:
                print('Fail to update tracker')
                break

            left, top, w, h = [int(v) for v in box]
            right = left + w
            bottom = top + h

            # save sizes of image
            top_bottom_list.append(np.array([top, bottom]))
            left_right_list.append(np.array([left, right]))

            # use recent 10 elements for crop (window_size=10)
            if len(top_bottom_list) > 10:
                del top_bottom_list[0]
                del left_right_list[0]

            # compute moving average
            avg_height_range = np.mean(top_bottom_list, axis=0).astype(np.int)
            avg_width_range = np.mean(left_right_list, axis=0).astype(np.int)
            avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)])  # (x, y)

            # compute scaled width and height
            avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
            avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

            # compute new scaled ROI
            avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
            avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])

            # fit to output aspect ratio
            if self.fit_to == 'width':
                avg_height_range = np.array([avg_center[1] - avg_width * size[1] / size[0] / 2,
                                             avg_center[1] + avg_width * size[1] / size[0] / 2]
                                            ).astype(np.int).clip(0, 9999)
                avg_width_range = avg_width_range.astype(np.int).clip(0, 9999)

            elif self.fit_to == 'height':
                avg_height_range = avg_height_range.astype(np.int).clip(0, 9999)
                avg_width_range = np.array([avg_center[0] - avg_height * size[0] / size[1] / 2,
                                            avg_center[0] + avg_height * size[0] / size[1] / 2]
                                           ).astype(np.int).clip(0, 9999)

            # crop image
            result_img = image[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

            # resize image to output size
            result_img = cv2.resize(result_img, size)

            # visualize
            pt1 = (int(left), int(top))
            pt2 = (int(right), int(bottom))
            cv2.rectangle(image, pt1, pt2, (255, 255, 255), 3)

            cv2.imshow('img', image)
            cv2.imshow('result', result_img)

            # write video
            if cv2.waitKey(1) == ord('q'):
                break

        # release everything
        self.cap.release()
        print('Press to exit')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test = TrackerHelper('kcf')
    img, roi = test.set_roi()
    test.run(img, roi)
