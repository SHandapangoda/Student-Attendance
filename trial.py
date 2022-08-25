

# thresholding using adaptive technique



import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter

def gauss():

    img = cv2.imread('sign-in-out-sheet-template-lovely-equipment-sign-out-sheet-template-beautiful-template-of-sign-in-out-sheet-template.jpg',0)
    #img = image.img_to_array(img, dtype='uint8')

    print(img.shape)
    ## output : (224,224,3)
    #plt.imshow(img_grey)

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    plt.figure(figsize=(20,10))
    plt.imshow(th3, cmap="gray")
    plt.show()

def mean():

    img = cv2.imread('sign-in-out-sheet-template-lovely-equipment-sign-out-sheet-template-beautiful-template-of-sign-in-out-sheet-template.jpg',0)
    #img = image.img_to_array(img, dtype='uint8')

    print(img.shape)
    ## output : (224,224,3)
    #plt.imshow(img_grey)

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    plt.figure(figsize=(20,10))
    plt.imshow(th3, cmap="gray")
    plt.show()

def blur():
    img = cv2.imread('sign-in-out-sheet-template-lovely-equipment-sign-out-sheet-template-beautiful-template-of-sign-in-out-sheet-template.jpg',0)
    #img = image.img_to_array(img, dtype='uint8')

    print(img.shape)
    ## output : (224,224,3)
    #plt.imshow(img_grey)

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    blur = cv2.medianBlur(th3,5)
    plt.figure(figsize=(20,10))
    plt.imshow(blur, cmap="gray")
    plt.show()


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def antiskew():
    img = cv2.imread('sign-in-out-sheet-template-lovely-equipment-sign-out-sheet-template-beautiful-template-of-sign-in-out-sheet-template.jpg',0)
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    
    scores = []
    for angle in angles:
        
        hist, score = find_score(img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle: {}'.format(best_angle))

    # correct skew
    data = inter.rotate(img, best_angle, reshape=False, order=0)

    img.save('skew_corrected.png')

def letters_extract(image_file):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        letter_crop = gray[y:y + h, x:x + w]       
        letters.append(letter_crop)
        cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255))
    
    cv2.namedWindow("win", cv2.WINDOW_FREERATIO)
    cv2.imshow("win",img)
    cv2.waitKey()

    return letters

letters_extract('/media/sithum/New Folder3/pythonProject/sign-in-out-sheet-template-lovely-equipment-sign-out-sheet-template-beautiful-template-of-sign-in-out-sheet-template.jpg')
