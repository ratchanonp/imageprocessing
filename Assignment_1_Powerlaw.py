import cv2
import numpy as np
import matplotlib.pyplot as plt


def power_law_transform(image_path, c, gamma):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_name = image_path.split("/")[-1].split(".")[0]

    image_float = image.astype(np.float32)
    image_transformed = c * (image_float ** gamma)
    image_normalize = (image_transformed - np.min(image_transformed)) / \
        (np.max(image_transformed) - np.min(image_transformed)) * 255
    gamma_corrected_img = image_normalize.astype(np.uint8)

    cv2.imwrite(
        f"./images/gamma_corrected_images/{image_name}-{c}-{gamma}.jpg", gamma_corrected_img)

    return gamma_corrected_img


images_list = ["cartoon.jpg", "scenery1.jpg", "scenery2.jpg"]

for image_path in images_list:
    path = f"images/{image_path}"

    fig = plt.figure(figsize=(7, 8))
    row, col = 4, 2

    # plot original image
    plt.subplot(row, col, 1)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Original Image")

    c_list = [0.5, 1, 2]
    gamma_list = [0.4, 2.5]

    for c in c_list:
        for gamma in gamma_list:
            plt.subplot(row, col, c_list.index(c) *
                        2 + gamma_list.index(gamma) + 3)
            plt.imshow(power_law_transform(path, c, gamma), cmap='gray')
            plt.axis('off')
            plt.title(f"c={c}, gamma={gamma}")

    plt.savefig(f"images/gamma_corrected_images/{image_path}")
    plt.show()
