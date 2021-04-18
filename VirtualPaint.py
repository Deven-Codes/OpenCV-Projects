import cv2
import numpy as np

# webcam capturing and display
cap = cv2.VideoCapture(0)
cap.set(3, 640) # defining width NOTE width have id 3
cap.set(4, 480) # defining height NOTE height have id 4
cap.set(10, 100) # setting brightness to 100 have id 10

# green : 27 80 45 100 255 255
# red : 0 100 25 3 255 255 
# blue : 60 230 100 122 255 255

myColors = [[27, 80, 45, 100, 255, 255], 
            [0, 100, 25, 3, 255, 255],
            [60, 230, 100, 122, 255, 255]]


myColorValues = [[140, 235, 134],
                 [18, 17, 128],
                 [212, 77, 23]]

myPoints = []        # [x, y, colorID]



# function for finding color
def findColor(img, myColors, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x,y = getContours(mask)
        cv2.circle(imgResult, (int(x),int(y)), 10, myColorValues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x,y, count])
        count += 1
    return newPoints
        



def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            # cv2.drawContours(imgResult, cnt, -1, (0, 0, 0), 5) # putting contour on image
            #curve length will help us approximate our shape
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x,y,w,h = cv2.boundingRect(approx)

    return x+w/2 , y

def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (int(point[0]),int(point[1])), 10, myColorValues[point[2]], cv2.FILLED)

if __name__ == "__main__":
    while True:
        success, img = cap.read()
        imgResult = img.copy()
        newPoints = findColor(img, myColors, myColorValues)
        if len(newPoints) != 0:
            for newP in newPoints:
                myPoints.append(newP)
        if len(myPoints) != 0:
            drawOnCanvas(myPoints, myColorValues)

        cv2.imshow("Result", imgResult)
        # cv2.imshow("output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break