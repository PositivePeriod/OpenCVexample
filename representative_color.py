import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv


class RepresentativeColor:
    def resize(self, image, x=0.1, y=0.1):
        image = cv2.resize(image, dsize=(0, 0), fx=x, fy=y, interpolation=cv2.INTER_AREA)
        return image

    def get_dominant_colors(self, image, n=3):
        t = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, colors = cv2.kmeans(pixels, n, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        print(f'Elapsed time : {time.time() - t}')
        return colors, counts

    def get_average_color(self, image):
        color = image.mean(axis=0).mean(axis=0)
        return color

    def show_colors(self, average, dominant, counts):
        avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)

        indices = np.argsort(counts)[::-1]
        freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
        rows = np.int_(img.shape[0] * freqs)

        dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(dominant[indices[i]])

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        ax0.imshow(avg_patch)
        ax0.set_title('Average color')
        ax0.axis('off')
        ax1.imshow(dom_patch)
        ax1.set_title('Dominant colors')
        ax1.axis('off')
        plt.show()


if __name__ == '__main__':
    test = RepresentativeColor()
    img = cv2.imread('./data/butterfly.jpg')
    import os
    if os.path.isfile('image_transformation.py'):
        import image_transformation
        helper = image_transformation.ImageTransform()
        img = helper.resize(img)
    cv2.imshow('Image', img)
    average_color = test.get_average_color(img)
    dominant_colors, counts = test.get_dominant_colors(img)
    test.show_colors(average_color, dominant_colors, counts)
