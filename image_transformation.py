import numpy as np
import cv2

# https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html
# https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220504966397&proxyReferer=https:%2F%2Fwww.google.com%2F


class ImageTransform:
    def resize(self, image, size=None, x=0.5, y=0.5):
        if size is None:
            # Relative size
            if x < 1 or y < 1:
                # Shrinking
                image_ = cv2.resize(image, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)
            else:
                # Zooming
                image_ = cv2.resize(image, None, fx=x, fy=y, interpolation=cv2.INTER_CUBIC)
        else:
            # Absolute size
            image_ = cv2.resize(image, dsize=size, interpolation=cv2.INTER_AREA)
        return image_

    def rotate(self, image, angle=45):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image_ = cv2.warpAffine(image, M, (w, h))
        return image_

    def quad_mask(self, image, points):
        # order of points -> clockwise; left top; right top; left bottom; right bottom to erase inside
        # point -> (width, height)
        h, w = image.shape[:2]

        black = np.zeros((h, w, 3), dtype=np.uint8)
        white = cv2.bitwise_not(black)

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32(points)

        M = cv2.getPerspectiveTransform(pts1, pts2)
        white_rotate = cv2.warpPerspective(white, M, (w, h))
        black_rotate = cv2.bitwise_not(white_rotate)
        img_filter = cv2.bitwise_and(black_rotate, image)
        return img_filter, black_rotate

    def circular_mask(self, image, center=None, radius=None):
        h, w = image.shape[:2]
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = dist_from_center <= radius
        black = np.zeros((h, w, 3), dtype=np.uint8)
        black[~mask] = 255  # change into white
        image_filter = cv2.bitwise_and(black, image)
        return image_filter, black

    def cover(self, image, mask, recover):
        un_mask = cv2.bitwise_not(mask)
        recover_filter = cv2.bitwise_and(un_mask, recover)
        img_filter = cv2.bitwise_or(image, recover_filter)
        return img_filter


if __name__ == '__main__':
    test = ImageTransform()
    img = cv2.imread('./data/bluff.jpg')

    img_resize = test.resize(img)
    cv2.imshow('Resize', img_resize)
    img_rotate = test.rotate(img_resize)
    cv2.imshow('Rotate', img_rotate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_mask1, quad_mask1 = test.quad_mask(img_resize, [[700, 110], [620, 530], [390, 60], [360, 380]])
    cv2.imshow('Erased1', img_mask1)
    cv2.imshow('Quad mask', quad_mask1)
    img_mask2, quad_mask2 = test.quad_mask(img_resize, [[620, 530], [390, 60], [360, 380], [700, 110]])
    cv2.imshow('Erased2', img_mask2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img2 = cv2.imread('./data/fruit.jpg')
    height, width = img_mask1.shape[:2]
    img2_resize = test.resize(img2, size=(width, height))
    image_cover1 = test.cover(img_mask1, quad_mask1, img2_resize)
    cv2.imshow('Cover1', image_cover1)

    img2_mask, circular_mask = test.circular_mask(img2_resize, center=(width / 4, height / 4))
    cv2.imshow('Circular mask', circular_mask)
    cv2.imshow('Circular mask', img2_mask)
    image_cover2 = test.cover(img2_mask, circular_mask, img_mask1)
    cv2.imshow('Cover2', image_cover2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
