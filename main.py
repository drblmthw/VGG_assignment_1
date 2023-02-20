import cv2 as cv
import numpy as np


def main():

    cv.namedWindow('W1', cv.WINDOW_AUTOSIZE)

    image = cv.imread('./img/TCGA-18-5592-01Z-00-DX1.tif')

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)

    dilated = cv.dilate(thresh, kernel, iterations=2)

    cv.bitwise_not(dilated, dilated)

    blur = cv.GaussianBlur(dilated, (11, 11), 0)

    canny = cv.Canny(blur, 30, 150, 3)

    d_canny = cv.dilate(canny, kernel, iterations=1)

    contours, hierarchy = cv.findContours(d_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cv.drawContours(image, contours, -1, (255, 22, 13), 7)

    cv.imshow('W1', image)

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()


