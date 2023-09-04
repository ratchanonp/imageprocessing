import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(image, gray_level):
    global image_name

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

     # make it scaled to gray_level
    gray_image = np.floor(gray_image / (256 // gray_level)) * (256 // gray_level)
    
    cv2.imwrite(f"./images/quantized_images/{image_name}_gray_{gray_level}.jpg", gray_image)
    
    return gray_image

images_list = ["flower.jpg", "fractal.jpeg", "fruit.jpg"]
gray_levels = [8, 16, 64, 128, 256]

image_name = ""

# for image_name in images_list:
for image_path in images_list:

    image_name = image_path.split(".")[0]

    # plot configuration
    rows, columns = 1, len(gray_levels) + 1
    fig = plt.figure(figsize=(30, 4))
    fig.suptitle(f"Image: {image_path}", fontsize=15, y=1.05)

    # read image
    image = cv2.imread(f"images/{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB

    # plot original image
    plt.subplot(rows, columns, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Original Image")

    for i in range(len(gray_levels)):
        gray_level = gray_levels[i]
        grayscale_img = display_image(image, gray_level)

        # plot quantized image
        plt.subplot(rows, columns, i + 2)
        plt.imshow(grayscale_img, cmap='gray')
        plt.axis('off')
        plt.title(f"Gray Level: {gray_level}")
    
    plt.savefig(f"images/quantized_images/{image_path}")
    plt.show()