import cv2
import numpy as np

image = cv2.imread('test1.png')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
cv2.imshow('blur image',blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('binary image',binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
cv2.imshow('Sharp image',sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

skeletonized_image = skeletonize(binary_image)
cv2.imshow('Skeleton Image',skeletonized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
