import numpy as np
import cv2
from matplotlib import pyplot as plt

def descriptor(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(128,128))
    
    convertedgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    convertedhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv",convertedhsv)

    lowerBoundary = np.array([0,40,30],dtype="uint8")#hsv ranges for skin
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(convertedhsv, lowerBoundary, upperBoundary)
    #cv2.imshow("inrange",skinMask)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    #cv2.imshow("masked",skinMask)
    
    skinMask = cv2.medianBlur(skinMask, 5)#remove noise
    skin = cv2.bitwise_and(convertedgray, convertedgray, mask = skinMask)#mask out hand

    img2 = cv2.Canny(skin,60,60)
    brisk = cv2.BRISK_create()
    #brisk = cv2.xfeatures2d.SURF_create()
    img2 = cv2.resize(img2,(256,256))
    kp, des = brisk.detectAndCompute(img2,None)
    img2 = cv2.drawKeypoints(img2,kp,None,(0,0,255),4)
    #plt.imshow(img2),plt.show()
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print(len(des))
    return des


#print(descriptor('001.jpg').shape)
