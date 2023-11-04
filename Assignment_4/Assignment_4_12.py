import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import Grid

# Read image
flower = cv.imread('./img/flower1.jpg', cv.IMREAD_GRAYSCALE)
fruit = cv.imread('./img/fruit.jpg', cv.IMREAD_GRAYSCALE)


def spatial_to_freq(img):
    # Calculate the 2D discrete Fourier Transform
    f = np.fft.fft2(img)
    # Shift the zero-frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)

    return fshift


def freq_to_spatial(freq_img):
    # Shift the zero-frequency component to the center of the spectrum
    f_ishift = np.fft.ifftshift(freq_img)
    # Calculate the 2D discrete Fourier Transform
    img_back = np.fft.ifft2(f_ishift)

    return img_back


def calculate_magnitude_spectrum(img):
    np.seterr(divide='ignore')
    return 20 * np.log(np.abs(img))


def create_notch_mask(height, width, radius=10, invert=False):
    mask = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            if np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2) <= radius:
                mask[i, j] = 1

    if invert:
        mask = 1 - mask

    return mask


def create_gaussian_mask(height, width, D0=10, invert=False):
    mask = np.zeros((height, width), np.uint8)

    # Change to float
    mask = np.float32(mask)

    cX, cY = width // 2, height // 2

    for i in range(-cY, cY):
        for j in range(-cX, cX):
            distance = np.sqrt(i ** 2 + j ** 2)
            mask[i + cY, j + cX] = np.exp(-(distance ** 2) / (2 * (D0 ** 2)))

    if invert:
        mask = 1 - mask

    return mask


def apply_mask(img, mask):
    return img * mask


def plot_3D(mask, title="", img_name=""):
    width, height = mask.shape

    fig = plt.figure()
    ax = Axes3D(fig)

    fig.add_axes(ax)

    X = np.arange(0, width, 1)
    Y = np.arange(0, height, 1)
    X, Y = np.meshgrid(X, Y)
    Z = mask[X, Y]

    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.title(title)
    plt.savefig(f"out/{img_name}/{title}.png")

    ax.remove()
    plt.close()


def main(img, img_name):

    freq_img = spatial_to_freq(img)
    width, height = freq_img.shape[0], freq_img.shape[1]

    # Radius
    radius_list = [10, 50, 100]

    # Cut-off D0
    D0_list = [10, 50, 100]

    # Notch Mask
    low_pass_mask_list = [{
        "mask": create_notch_mask(width, height, radius),
        "title": f"Low Pass Notch Mask with radius {radius}"
    } for radius in radius_list]

    high_pass_mask_list = [{
        "mask": create_notch_mask(width, height, radius, invert=True),
        "title": f"High Pass Notch Mask with radius {radius}"
    } for radius in radius_list]

    # Gaussian Mask
    low_pass_gaussian_mask_list = [{
        "mask": create_gaussian_mask(width, height, D0),
        "title": f"Low Pass Gaussian Mask with D0 {D0}"
    } for D0 in D0_list]

    high_pass_gaussian_mask_list = [{
        "mask": create_gaussian_mask(width, height, D0, invert=True),
        "title": f"High Pass Gaussian Mask with D0 {D0}"
    } for D0 in D0_list]

    # Concatenate all masks
    masks = low_pass_mask_list + high_pass_mask_list + low_pass_gaussian_mask_list + high_pass_gaussian_mask_list

    # Calculate the magnitude spectrum
    magnitude_spectrum = calculate_magnitude_spectrum(freq_img)
    plot_3D(magnitude_spectrum, title="Magnitude Spectrum", img_name=img_name)

    # Plot the magnitude spectrum
    plt.subplot(121)
    plt.title('Input Image')
    plt.imshow(img, cmap='gray')

    plt.subplot(122)
    plt.title('Magnitude Spectrum')
    plt.imshow(magnitude_spectrum, cmap='gray')

    plt.savefig(f"out/{img_name}/magnitude_spectrum.png")

    img_ratio = width / height

    for mask_dict in masks:

        mask = mask_dict["mask"]
        title = mask_dict["title"]

        plot_3D(mask, title=title, img_name=img_name)

        result = apply_mask(freq_img, mask)

        img_result = freq_to_spatial(result)

        fig, ax = plt.subplot_mosaic([
            ["output", "magnitude"]
        ], figsize=(
            10, 10 * img_ratio
        ))

        # Plot the magnitude spectrum after applying the mask
        ax["output"].set_title('Output Image')
        ax["output"].imshow(img_result.real, cmap='gray')

        ax["magnitude"].set_title('Magnitude Spectrum')
        ax["magnitude"].imshow(calculate_magnitude_spectrum(result), cmap='gray')

        fig.suptitle("Applied " + title, size=20, y=0.85)
        fig.tight_layout()

        fig.savefig(f"out/{img_name}/Applied {title}.png")

        plt.close()


main(fruit, "fruit")
main(flower, "flower")
