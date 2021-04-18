import cv2
import numpy as np

widthImg = 640
heightImg = 480


# webcam capturing and display
cap = cv2.VideoCapture(0)
cap.set(3, widthImg) # defining width, width have id 3
cap.set(4, heightImg) # defining height, height have id 4
cap.set(10, 100) # setting brightness to 100 have id 10


# prepocessing the image edges
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDilate = cv2.dilate(imgCanny, kernel, iterations = 2)
    imgThreshold = cv2.erode(imgDilate, kernel, iterations = 1)

    return imgThreshold

# get the biggest contour fron the image
def getContours(img):
    biggest = np.array([]) # points of biggest contour
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (0,   255, 0), 3) # putting contour on image
            #curve length will help us approximate our shape
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (0,   255, 0), 20) # print the biggest contour points

    return biggest  

def reorder(myPoints):
   myPoints = myPoints.reshape((4,2))
   myPointsNew = np.zeros((4,1,2), np.int32)
   add = myPoints.sum(1)
   
   myPointsNew[0] = myPoints[np.argmin(add)]
   myPointsNew[3] = myPoints[np.argmax(add)]

   diff = np.diff(myPoints, axis=1)
   myPointsNew[1] = myPoints[np.argmin(diff)]
   myPointsNew[2] = myPoints[np.argmax(diff)]

   return myPointsNew
    


# wrap the image into birds eye view
def getWarp(img, biggest):

#  [[[  7 168]] [[  0 324]] [[590 328]] [[599 189]]]
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped,(widthImg, heightImg))

    return imgCropped

if __name__ == "__main__":
    while True:
        success, img = cap.read()
        cv2.resize(img, (widthImg, heightImg))
        imgContour = img.copy()

        imgThreshold = preProcessing(img)
        biggest = getContours(imgThreshold)

        if biggest.size != 0:
            imgWarpped = getWarp(img, biggest)
        else:
            imgWarpped = img.copy()
            
        cv2.imshow("Output", imgWarpped)    
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break