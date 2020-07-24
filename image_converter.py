import cv2


class ImageConverter:
    def canny(self, image, low=50, high=200):
        result = cv2.Canny(image, low, high)
        return result

    def detail(self, image, sigma_s=10, sigma_r=0.15):
        result = cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)
        return result


if __name__ == '__main__':
    test = ImageConverter()
    img = cv2.imread('./data/butterfly.jpg')
    import os
    if os.path.isfile('image_transformation.py'):
        import image_transformation
        helper = image_transformation.ImageTransform()
        img = helper.resize(img)
    cv2.imshow('Original', img)
    canny = test.canny(img)
    cv2.imshow('Canny', canny)
    detail = test.detail(img)
    cv2.imshow('Detail', detail)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
