import cv2
import matplotlib.pyplot as plt

def enhance_img(image_path):

    image = cv2.imread(f"images/{image_path}")
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    L = 256

    for i in range(len(grayscale_img)):
        for j in range(len(grayscale_img[0])):
            if grayscale_img[i][j] <= L / 3:
                grayscale_img[i][j] = 5 * L / 6
            elif grayscale_img[i][j] <= 2 * L / 3:
                grayscale_img[i][j] = (-2 * grayscale_img[i][j]) + 384
            else:
                grayscale_img[i][j] = L / 6

    cv2.imwrite(f"./images/enhanced_images/{image_path}", grayscale_img)

    return grayscale_img

rows, columns = 1, 2
images_list = ["flower.jpg", "traffic.jpg", "tram.jpg"]

for image_path in images_list:
    fig = plt.figure(figsize=(30, 3))
    enhance_img(image_path)

    # plot original image
    plt.subplot(rows, columns, 1)

    image = cv2.imread(f"images/{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Original Image")

    # plot enhanced image
    plt.subplot(rows, columns, 2)
    plt.imshow(cv2.imread(f"images/enhanced_images/{image_path}"), cmap='gray')
    plt.axis('off')
    plt.title(f"Enhanced Image")

    plt.show()
