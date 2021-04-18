import cv2
import time

first_frame = None


video = cv2.VideoCapture(0)

while True:
    check, frame=video.read()

    # GaussianBlur() will blur the frame gray is the frame, (21,21) is height and width of gaussian kernel, std deviation is 0.
    # The Gaussian filter is a low-pass filter that removes the high-frequency components
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    # If first_frame has None in it then the first frame of the video (numpy array) will store in variable (first_frame)
    # and rest of the code will not be executed due to continue command but while loop will initialise the next frame.
    # on the second time first_frame will have a numpy array stored in it so this If statement will not execute again,

    if first_frame is None:
        first_frame = gray
        continue

    # absdiff() is method for absolute difference between two frames.
    delta_frame=cv2.absdiff(first_frame,gray)
    
    # delta_frame is difference of intensity of pixels at different time due to movement in the video.
    # In here 30 is the delta frame which is the threshold limit, if delta frame exceeeds this value white pixel will be shown
    # but in this cv2.THRESH_BINARY only requires delta frame that is why [1] is passed.
    # there are other type of threshold methods but we are using cv2.THRESH_BINARY.
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    # Erosion and dilation are morphological image processing operations. 
    # Morphological image processing basically deals with modifying geometric structures in the image.
    # These operations are primarily defined for binary images, but we can also use them on grayscale images. 
    # Erosion basically strips out the outermost layer of pixels in a structure, where as dilation adds 
    # an extra layer of pixels on a structure.
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)
    
    # Movement in the video is detected and are store in list contours 
    # a rectange is used to highlight the area.
    contours, _= cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour)<2500:
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    cv2.imshow("Color Frame",frame)
    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)


    key = cv2.waitKey(20)
    print(gray)
    print(delta_frame)

    # The ord() method returns an integer representing Unicode code (ASCII code) point for the given Unicode character.
    # For example 'A'=65, 'a'=97 , '$'=36 etc.
    if key == ord('q'):
        break


# for accessing the camera or video .release() method is used.
video.release()
cv2.destroyAllWindows()