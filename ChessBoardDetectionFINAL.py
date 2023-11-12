from asyncio.windows_events import NULL
import cv2
import numpy as np
from math import pi



#Crop the image for deleting the outside part of the table and optimizing the chessboard recognition.
#The number below means the number of pixels to delete from top to bottom. Put 0 to disable the function.
#Put attenction to no delete parts of pieces on top of Chessboard.
DistanceUntilTable = 0

#Rise the height of the found squares because of the height of pieces. Put 0 to disable the function.
HeightPieces=30

#Shape of Warped Image obtained. It will then be increased by 'HeightPieces' in order not to cut the head off the pieces.
widthWarpedImage = 300
heightWarpedImage = 300







original_img ='3_Color.png'
# original_img ='01.jpg'
# original_img = cv2.VideoCapture(0) # for using CAM
# original_img ='newEnvs/11_Color.png'







#Crop first part of the image
def CropImage(image):
    global DistanceUntilTable
    # Cropping an image
    cropped_image = image[DistanceUntilTable:image.shape[0], 0:image.shape[1]]
    return cropped_image 


#Find only vertical and horizontal lines with Hough T.
def VerHorLines(lines):
    rad=0.01
    out = []
    for line in range(len(lines)):
        if (lines[line][0][1]<=rad or lines[line][0][1]>=(2*pi)-rad or (lines[line][0][1]<=(pi/2)+rad and lines[line][0][1]>=(pi/2)-rad)):
            out.append(lines[line])
    return out



"""
So if line is passing below the origin, it will have a positive rho and angle less than 180.
If it is going above the origin, instead of taking angle greater than 180, angle is taken less than 180, and rho is taken negative.
Any vertical line will have 0 degree and horizontal lines will have 90 degree.
"""
#Function to resolve the negative rho in order to compare lines.
def SolveHoughProblem(lines):
    for line in lines:
        if (line[0][0]<0):
          line[0][0]=-line[0][0]
          line[0][1]=line[0][1]+np.pi
    return lines


#Restore the Hough L.T. format with negative rho
def InvertedSolveHoughProblem(lines):
    for line in lines:
        if (line[0][1]>pi):
          line[0][0]=-line[0][0]
          line[0][1]=line[0][1]-np.pi
    return lines


#Average of similar lines
def CentralLine(lines):
    rho=0
    theta=0
    for line in lines:
        rho=rho+line[0][0]
        theta=theta+line[0][1]
    rho=rho/len(lines)
    theta=theta/len(lines)
    line = [[rho,theta]]
    return line


#It detects similar lines
def DetectSimilarLines(lines):
    similar = False
    rad=0.01
    distance=20
    out = []
    temp = []
    checkMatrix = np.full(len(lines), False)
    for i in range(0,len(lines)):
        if (checkMatrix[i]==False):
            checkMatrix[i]=True
            temp.append(lines[i])
            for j in range(i+1,len(lines)):
                if (checkMatrix[j]==False):
                    if (abs(lines[i][0][1]-lines[j][0][1])<=rad and abs(lines[i][0][0]-lines[j][0][0])<=distance):
                        checkMatrix[j]=True
                        temp.append(lines[j])
                        similar = True
            if similar:
                out.append(CentralLine(temp))
            else:
                out.append(temp)
            temp = []
    return out


#Find the 4 main lines at the ends
def FourMainLines(out,lines):
    vertLines = []
    horLines = []
    rad=0.1
    for line in range(len(lines)):
        if (lines[line][0][1]<=rad or lines[line][0][1]>=(2*pi)-rad):
            vertLines.append(lines[line])
        if (lines[line][0][1]<=(pi/2)+rad and lines[line][0][1]>=(pi/2)-rad):
            horLines.append(lines[line])
    out.append(vertLines[0])
    out.append(vertLines[0])
    out.append(horLines[0])
    out.append(horLines[0])
    for line in range(len(vertLines)):
        if (vertLines[line][0][0]<out[0][0][0]):
            out[0]=vertLines[line]
        if (vertLines[line][0][0]>out[1][0][0]):
            out[1]=vertLines[line]
    for line in range(len(horLines)):
        if (horLines[line][0][0]<out[2][0][0]):
            out[2]=horLines[line]
        if (horLines[line][0][0]>out[3][0][0]):
            out[3]=horLines[line]
    out[0][0][1]=0
    out[1][0][1]=0
    out[2][0][1]=round(pi/2, 2)
    out[3][0][1]=round(pi/2, 2)
    return out


#Reconstructs the grid with precision
def RebuildGrid(lines):
    out = []
    out = FourMainLines(out,lines)
    VertDist=out[1][0][0]-out[0][0][0]
    HorDist=out[3][0][0]-out[2][0][0]
    VertSqr=VertDist/8
    HorSqr=HorDist/8
    temp1=out[0][0][0]
    x=0
    while (x<7):
        temp1+=VertSqr
        temp = [[temp1,0]]
        out.append(temp)
        x+=1
    temp1=out[2][0][0]
    x=0
    while (x<7):
        temp1+=HorSqr
        temp = [[temp1,1.57]]
        out.append(temp)
        x+=1
    return out


#Find the detected object with the biggest area
def findChessBoard(shape):
    area = NULL
    x=NULL
    for i in range(len(shape)):
        if (area==NULL):
            area = cv2.contourArea(shape[i])
            x=i
        else:
            if (cv2.contourArea(shape[i])>area):
                area = cv2.contourArea(shape[i])
                x=i
    return x


#Detect the Top Left Corner of the Chessboard
def tleft(array):
    global HeightPieces
    for i in range(len(array)):   
        if (i==0):
            x=array[i][0]
        else:
            if ((array[i][0][0]+array[i][0][1])<(x[0]+x[1])):
                x=array[i][0]
    return x


#Detect the Top Right Corner of the Chessboard
def tright(array):
    global HeightPieces
    for i in range(len(array)):   
        if (i==0):
            x=array[i][0]
        else:
            if ((array[i][0][0]-array[i][0][1])>(x[0]-x[1])):
                x=array[i][0]
    return x


#Detect the Bottom Left Corner of the Chessboard
def bleft(array):
    for i in range(len(array)):   
        if (i==0):
            x=array[i][0]
        else:
            if ((array[i][0][0]-array[i][0][1])<(x[0]-x[1])):
                x=array[i][0]
    return x


#Detect the Bottom Right Corner of the Chessboard
def bright(array):
    for i in range(len(array)):   
        if (i==0):
            x = array[i][0]
        else:
            if ((array[i][0][0]+array[i][0][1])>(x[0]+x[1])):
                x = array[i][0]
    return x


#Intersection between lines
def FindIntersections(image,lines):
    intersections = []
    for x in range(0,len(lines)):
       for y in range(x+1,len(lines)):
           rho1, theta1 = lines[x][0]
           rho2, theta2 = lines[y][0]
           A = np.array([[np.cos(theta1), np.sin(theta1)],
                         [np.cos(theta2), np.sin(theta2)]])
           B = np.array([[rho1], [rho2]])
           if (np.linalg.det(A)!=0): #check if the matrix is invertible
               x0, y0 = np.linalg.solve(A, B)
               if ((x0>=0 and x0<=image.shape[0]) and (y0>=0 and y0<=image.shape[1])):
                   intersection = int(np.round(x0)), int(np.round(y0))
                   intersections.append(intersection)
    return intersections


#Draw intersections among lines
def DrawIntersections(image, intersections):
    for i in range(0,len(intersections)):
        x = int(intersections[i][0]) #cv2.circle wants integer variables
        y = int(intersections[i][1])
        cv2.circle(image, (x, y), 5, (0,255,0), 1)


#Draw lines on an image
def DrawLines(image, lines):
    for line in range(len(lines)):
        rho,theta=lines[line][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),1)


#Look for near points to build squares. It is needed by the following functions FindSquares()
def FindNearPoint(intersections,typePoint):
    min = intersections[0]  
    if (typePoint==0):
        for x in range(1,len(intersections)):
            if (intersections[x][0]<min[0]):
                min = intersections[x]
    if (typePoint==1):
        min = intersections[0]    
        for x in range(1,len(intersections)):
            if (intersections[x][1]<min[1]):
                min = intersections[x]
    return min


#Find all squares among interserctions and sort them
def FindSquares(intersections):
    squares = []
    for x in range(len(intersections)):
        a0 = intersections[x]
        a1 = 0
        a2 = 0
        a3 = 0
        temp = []
        for y in range(len(intersections)):
            # if (intersections[y][1]==a0[1] and intersections[y][0]>a0[0]):
            if (abs(intersections[y][1]-a0[1])<10 and intersections[y][0]>a0[0]):
                temp.append(intersections[y])
        if (len(temp)!=0):
            a1 = FindNearPoint(temp,typePoint=0)
            temp = []
            for z in range(len(intersections)):
                # if (intersections[z][0]==a0[0] and intersections[z][1]>a0[1]):
                if (abs(intersections[z][0]-a0[0])<10 and intersections[z][1]>a0[1]):
                    temp.append(intersections[z])
            if (len(temp)!=0):
                a2 = FindNearPoint(temp,typePoint=1)
                a3 = (a1[0],a2[1])
                temp = [a0,a1,a2,a3]
                squares.append(temp)
    squares.sort(key=lambda x: (x[0][0], x[0][1]))
    return squares


#Draw squares on an image
def DrawSquares(image, squares):
    for x in range(len(squares)):
        cv2.rectangle(image,squares[x][0],squares[x][3],(0,255,0),1)


#Save each square as an image
def SaveImagesFromSquares(image,squares):
    global HeightPieces
    i=0
    for x in range(ord('a'),ord('h')+1):
        for y in range(1,9):
            if (squares[i][0][1]-HeightPieces>=0):
                cropped_img = warped_img2[squares[i][0][1]-HeightPieces:squares[i][2][1], squares[i][0][0]:squares[i][1][0]]
            else: cropped_img = warped_img2[0:squares[i][2][1], squares[i][0][0]:squares[i][1][0]]
            cv2.imwrite("Squares/"+chr(x)+str(y)+".jpg", cropped_img)
            # cv2.imshow('a', cropped_img)
            # cv2.waitKey(0)
            i+=1


image = cv2.imread(original_img)

if (DistanceUntilTable>0):
    image = CropImage(image)

image2 = image.copy()

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, img_thr = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY)
contours, hierarchy  = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[findChessBoard(contours)]

cv2.drawContours(image, contour, -1, (0,255,75), 2)

BRcorner = bright(contour)
TRcorner = tright(contour)
BLcorner = bleft(contour)
TLcorner = tleft(contour)

cv2.circle(image, BRcorner, radius=5, color=(0, 0, 255), thickness=-1)
cv2.circle(image, TRcorner, radius=5, color=(0, 0, 255), thickness=-1)
cv2.circle(image, BLcorner, radius=5, color=(0, 0, 255), thickness=-1)
cv2.circle(image, TLcorner, radius=5, color=(0, 0, 255), thickness=-1)

if (TLcorner[1]-HeightPieces>0 and TRcorner[1]-HeightPieces>0):
    rect = np.array(((TLcorner[0]-5, TLcorner[1]-HeightPieces), (TRcorner[0]+5, TRcorner[1]-HeightPieces), (BRcorner[0]+5, BRcorner[1]+5),(BLcorner[0]-5, BLcorner[1]+5)), dtype="float32")
else:   rect = np.array(((TLcorner[0]-5, TLcorner[1]-5), (TRcorner[0]+5, TRcorner[1]-5), (BRcorner[0]+5, BRcorner[1]+5),(BLcorner[0]-5, BLcorner[1]+5)), dtype="float32")


dst = np.array([[0,0], [widthWarpedImage-1,0], [widthWarpedImage-1,heightWarpedImage-1], [0,heightWarpedImage-1]],dtype="float32")
M = cv2.getPerspectiveTransform(rect,dst)
warped_img = cv2.warpPerspective(image2, M, (widthWarpedImage, heightWarpedImage))
warped_img2 = warped_img.copy()

canny_img = cv2.Canny(warped_img, 100, 150)

# Hough Lines T.
DistanceDrawLines = int(((widthWarpedImage+heightWarpedImage)/2) * 15/100)
lines = cv2.HoughLines(canny_img, 1, np.pi/180, DistanceDrawLines)

if lines is not None:
    lines = SolveHoughProblem(lines)

    lines = VerHorLines(lines)

    lines = DetectSimilarLines(lines)
    
    lines = RebuildGrid(lines)
    # print(len(lines))
    
    lines = InvertedSolveHoughProblem(lines)

    intersections = FindIntersections(warped_img,lines)
    # print(intersections)
    
    DrawIntersections(warped_img, intersections)

    canny_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)

    DrawLines(canny_img,lines)

    squares = FindSquares(intersections)
    # print(squares)
    print(len(squares))

    DrawSquares(warped_img, squares)

    SaveImagesFromSquares(warped_img2, squares)
    


# wimage_gray = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Gray', img_gray)
# cv2.imshow('Thr', img_thr)
cv2.imshow('Contours', image)
cv2.imshow('W_img', warped_img)
cv2.imshow('W_img2', warped_img2)
# cv2.imshow('W_img3', wimage_gray)
# cv2.imshow('crpi', cropped_img)


cv2.imshow('canny', canny_img)

# cv2.imwrite("contours_"+original_img,canny_img)
# cv2.imwrite("contours1_"+original_img,warped_img)
# cv2.imwrite("warped_1"+original_img,wimage_gray)
# cv2.imwrite("grey"+original_img,img_gray)
# cv2.imwrite("threshold"+original_img,img_thr)
# cv2.imwrite("warped"+original_img,warped_img2)


cv2.waitKey(0)