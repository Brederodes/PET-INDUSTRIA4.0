import numpy as np
import cv2 as cv

def analyse():
    return

topAreaHeight, bottomAreaHeight= (20,20)
#valor de diferenca na deteccao de diferenca
thresholdValue= 90
#valores maximos de area e perimetro
maxArea= 300
maxPerimeter= 150

cam= cv.VideoCapture(0)

if not cam.isOpened():
    print("no camera")
    exit()

ret, referenceFrame = cam.read()
height, width = referenceFrame.shape[:2]

grayReferenceFrame = cv.cvtColor(referenceFrame, cv.COLOR_BGR2GRAY)
grayBlurredReferenceFrame = cv.GaussianBlur(grayReferenceFrame,(5,5),0)

topProcessedReferenceFrame= grayBlurredReferenceFrame.copy()[0:topAreaHeight, 0:width]
bottomProcessedReferenceFrame= grayBlurredReferenceFrame.copy()[height-bottomAreaHeight:height, 0:width]


isRecording= False
beltObjectCount= 0

topHasObject= False
topHadObject= False
bottomHasObject= False
bottomHadObject= False

while True:
    ret, frame = cam.read()
    if not ret:
        print("no image")
        break
        
    topFrame= frame.copy()[0:topAreaHeight, 0:width]
    bottomFrame= frame.copy()[height-bottomAreaHeight:height, 0:width]
    
    topGrayFrame= cv.cvtColor(topFrame, cv.COLOR_BGR2GRAY)
    bottomGrayFrame= cv.cvtColor(bottomFrame, cv.COLOR_BGR2GRAY)
    
    topProcessedFrame= cv.GaussianBlur(topGrayFrame, (5,5), 0)
    bottomProcessedFrame= cv.GaussianBlur(bottomGrayFrame, (5,5), 0)
    
    topDifferenceImage= cv.absdiff(topProcessedReferenceFrame,topProcessedFrame)
    bottomDifferenceImage= cv.absdiff(bottomProcessedReferenceFrame,bottomProcessedFrame)
        
    ret,topDifferenceArea= cv.threshold(topDifferenceImage, thresholdValue,255,cv.THRESH_BINARY)
    ret,bottomDifferenceArea= cv.threshold(bottomDifferenceImage, thresholdValue,255,cv.THRESH_BINARY)
    topAreaCount= cv.countNonZero(topDifferenceArea)
    bottomAreaCount= cv.countNonZero(bottomDifferenceArea)
    
    topPerimeterImage= cv.Laplacian(topDifferenceArea, cv.CV_8U)
    bottomPerimeterImage= cv.Laplacian(bottomDifferenceArea, cv.CV_8U)
    topPerimeterCount= cv.countNonZero(topPerimeterImage)
    bottomPerimeterCount= cv.countNonZero(bottomPerimeterImage)
    
    
    cv.imshow('live',frame)
    cv.imshow('TDA',topDifferenceArea)
    cv.imshow('BDA',bottomDifferenceArea)
    
    #print("topArea:",topAreaCount," |topPerimeter:",topPerimeterCount,"\n")
    #print("bottomArea:",bottomAreaCount," | bottomPerimeter:",bottomPerimeterCount,"\n")

    topHasObject= (topAreaCount>maxArea and topPerimeterCount<maxPerimeter)
    bottomHasObject= (bottomAreaCount>maxArea and bottomPerimeterCount<maxPerimeter)
    
    if topHasObject: 
        if not topHadObject:
            beltObjectCount += 1
            topHadObject= True
    else:
        topHadObject= False
    
    
    if bottomHasObject:
        bottomHadObject= True
    else:
        if bottomHadObject:
            beltObjectCount -= 1
            bottomHadObject= False
            
    isRecording= (beltObjectCount>0)
    if isRecording:
        analyse() 
    print("\nOBJECT COUNT:",beltObjectCount)
    
    #apertar q para sair
    if cv.waitKey(100)== ord('q'):
        break
