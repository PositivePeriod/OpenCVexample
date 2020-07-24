import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter


# https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html

def find_ellipse(image):
    image_gray = color.rgb2gray(image)
    edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
    # The accuracy -> the bin size of a major axis
    result.sort(order='accumulator')
    return list(result[-1]), edges


def draw_ellipse(image, ellipse, edges):
    yc, xc, a, b = [int(round(x)) for x in ellipse[1:5]]
    orientation = ellipse[5]
    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex='all', sharey='all')
    ax1.set_title('Original')
    ax1.imshow(image_rgb)
    ax2.set_title('Edge with white / Result with red')
    ax2.imshow(edges)
    plt.show()


if __name__ == '__main__':
    image_rgb = data.coffee()[0:220, 160:420]
    ellipse, edges = find_ellipse(image_rgb)
    draw_ellipse(image_rgb, ellipse, edges)