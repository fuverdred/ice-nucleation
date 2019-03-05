import os
import re
import datetime

import cv2
import matplotlib.pyplot as plt
import xlwt#excel

class Droplet:
    """
    Contains the location, size and intensity information of each droplet
    """
    def __init__(self, c, r,img):
        self.i = int(c[0])
        self.j = int(c[1])
        self.r = int(r)
        self.area = r*r#Don't care about pi, just for normalising intensity change with area
        self.intensity = []#normalised to area
        self.freeze_temperature = None#Becomes a number if droplet freezes
        self.find_pixels()
        self.intensity = [self.find_intensity(img)]
        self.norm_intensity = [1.0]#Normalised to initial intensity so first is 1

    def find_pixels(self):
        #Find all pixel co-ords within the circle. (find one quadrant then mirror twice)
        pixels = []
        for x in range(self.r+1):
            Y = int((self.r**2-x*x)**0.5)#Pythag
            for y in range(Y+1):
                pixels.append([y,x])
        pixels += [[i,-j] for i,j in pixels]#mirror in x=0
        pixels += [[-i,j] for i,j in pixels]#mirror in y=0
        pixels = [[i+self.i,j+self.j] for i,j in pixels]#translate to circle centre, note reversal
        self.pixels = pixels

    def find_intensity(self, img):
        #Get the intensity normalised to droplet area
        return sum([img[j][i] for i,j in self.pixels])/self.area

    def check_frozen(self, temperature, img, threshold = 15):
        #Compare previous intensity to current
        self.intensity.append(self.find_intensity(img))
        self.norm_intensity.append(self.intensity[-1]/self.intensity[0])
        if self.intensity[-1] - self.intensity[0] > threshold and not self.freeze_temperature:
            self.freeze_temperature = temperature

    def draw(self, img, colour = (0,0,255), thickness = 4):
        cv2.circle(img,(self.i,self.j), self.r, colour, thickness)

def check_border(x,y,r,img, border = 10):
    """
    Check if droplet is hanging over the edge of the picture. input
    image must be single channel.
    """
    Y,X = img.shape
    if x-r < border or \
       x+r > X-border or \
       y-r < border or\
       y+r > Y-border:
        return False
    return True

def stabilise(img,x,y):
    """
    Takes in integer values for the image displacement. Moves the image in
    the opposite direction and adds a border for the cropped bit.
    """
    Y,X,_ = img.shape
    if x>=0 and y>=0:
        return cv2.copyMakeBorder(img[y:,x:], 0,y,0,x,cv2.BORDER_CONSTANT)
    if x<=0 and y<=0:
        return cv2.copyMakeBorder(img[:Y+y, :X+x], abs(y), 0, abs(x), 0, cv2.BORDER_CONSTANT)
    if x<=0 and y>=0:
        return cv2.copyMakeBorder(img[y:, :X+x], 0, y, abs(x), 0, cv2.BORDER_CONSTANT)
    if x>=0 and y<=0:
        return cv2.copyMakeBorder(img[:Y+y, x:], abs(y), 0, 0, x, cv2.BORDER_CONSTANT)

            
def tryint(s):#This is for sorting the files into human numerical order
    try: return int(s)
    except: return s

def alphanum_key(s):#This is for sorting the files into human numerical order
    return [tryint(c) for c in re.split("([0-9]+)", s)]

def make_video(files, droplets):
    print("Making Video")
    img = cv2.imread(files[0])
    out = cv2.VideoWriter(files[0][:-7]+".avi",cv2.VideoWriter_fourcc(*"XVID"), 10, img.shape[:2][::-1])
    for frame,file in enumerate(files,1):
        print("Adding frame ", frame)
        img = cv2.imread(file)
        for drop in droplets:
            if drop.freeze_frame < frame:
                cv2.circle(img, (drop.i,drop.j), drop.r+2, (0,255,0), 3)
            else:
                cv2.circle(img, (drop.i, drop.j), drop.r+2, (0,0,255),3)
        out.write(img)
    out.release()
    print("Finished")

def freezing_temps(droplets,start_frame, startT, dTdn = -0.25):
    """
    Return a list of the freezing temperature of each droplet, along with
    the radius of the droplet
    """
    return [[d.r, startT+((d.freeze_frame-start_frame)*DTdn)]
            for d in droplets if d.frozen]

def blob_detect(img):
    """
    Use cv2 simple blob detector to find droplets. img should already have been
    processed to make the droplets as clear as possible
    """
    drop_params = cv2.SimpleBlobDetector_Params()
    drop_params.filterByConvexity = False
    drop_params.minConvexity = 0.75
    drop_params.maxConvexity = 1.0
    drop_params.filterByArea = False
    drop_params.minArea = 1000#pixels
    drop_params.maxArea = 50000#pixels
    drop_params.filterByCircularity = False
    drop_params.minCircularity = 0.5
    drop_params.maxCircularity = 1.0#circle

    detector = cv2.SimpleBlobDetector_create(drop_params)
    keypoints = detector.detect(img)
    #return format tuple of ((x,y),r)
    return [((k.pt[0],k.pt[1]),k.size/2) for k in keypoints]

def circle_detect(img):
    """
    Use Hough circles to find droplets. img should already have been processed
    to make the droplets as clear as possible
    """
    circle_params = dict(dp = 2,#See cv2.HoughCircle docs
                         minDist = 50,
                         param2 = 50,
                         minRadius = 1,
                         maxRadius = 100)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, **circle_params)[0]
    #return format tuple of ((x,y),r)
    return [((c[0],c[1]),c[2]) for c in circles]

def draw_circles(circles, img):
    for c in circles:
        cv2.circle(img, (int(c[0][0]),int(c[0][1])), int(c[1]),(255,255,255),
                   10)
    return img

def method_1(img, threshold = 5):
    """
    Process image by looking at the red channel only
    """
    _,thr = cv2.threshold(img[:,:,2], threshold, 255, cv2.THRESH_BINARY)
    return thr

def method_2(img, threshold = 170, blockSize = 15, C=5):
    """
    Process image with adaptive threshold
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, blockSize, C)
    return thr

def method_3(img, threshold = 150):
    """
    Process image with straightforward threshold
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thr = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY)
    return thr

    
##CONSTANTS AND PARAMETERS#####################
file_extension = ".tiff"
border = 50#Number of pixels to ignore around the edge of the image
threshold = 12#change in brightness signifying freezing (arbitrary units)
frameRate = 2#for saved video

regex = re.compile(r"\d+")#For extracting file numbers
dTdn = -0.25#Change in temperature per frame
start_temp = 0#Temperature at first frame

feature_params = dict( maxCorners = 50,#Number of points to track
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)

lk_params = dict(winSize = (15,15),#Image flow parameters
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
###############################################

plt.ion()#interactive plotting

##Sort files##############
os.chdir("C:/Users/fc12293/Chrome Local Downloads/gary")
files = [i for i in os.listdir() if i[-len(file_extension):] == file_extension]
files.sort(key = alphanum_key)#Human sort of numbers 1->2->3 instead of 1->11->12...
##########################

##Setting up droplets#####
img = cv2.imread(files[0])
img = cv2.medianBlur(img, 5)
intensity = sum(sum(img[:,:,2]))
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thr = cv2.threshold(grey, 60, 255, cv2.THRESH_BINARY_INV)

p1 = cv2.goodFeaturesToTrack(grey, mask = None, **feature_params)#For droplet tracking
start_frame = int(regex.findall(files[0])[-1])#for working out temperature

test = method_1(img)
circles = circle_detect(test)
img = draw_circles(circles, img)
plt.imshow(img)

#Put the circle data into the droplet class, exclude any too near the edge.
droplets = [Droplet(*x,grey) for x in circles if check_border(*x,grey,border)]

##########################

##Finding Freezing########

#setup video file
out = cv2.VideoWriter(files[0][:-len(file_extension)]+".avi",cv2.VideoWriter_fourcc(*"XVID"), frameRate, img.shape[:2][::-1])

for file in files[2:]:#Already looked at the first one, shouldn't have any frozen
    print("Analysing ", file)
    img = cv2.imread(file)
    img = cv2.medianBlur(img, 5)
    if sum(sum(img[:,:,2]))-intensity > 10000:
       continue
    grey2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = int(regex.findall(file)[-1])
    temperature = start_temp+((frame-start_frame)*dTdn)
    print(f"Frame {frame} : Temperature {temperature}")
    
    p2,st,err = cv2.calcOpticalFlowPyrLK(grey,grey2, p1, None, **lk_params)#tracking
    movement = p2-p1#Take the difference
    x = int(sum([i[0][0] for i in movement])/len(movement))#average the x movement
    y = int(sum([i[0][1] for i in movement])/len(movement))#average the y movement
    print(f"Image movement (x,y):({x},{y})")
    img = stabilise(img, x, y)
    
    for drop in droplets:
        drop.check_frozen(temperature,grey2,threshold)
        if drop.freeze_temperature:
            drop.draw(img, (0,255,0))#draw in green
        else:
            drop.draw(img, (0,0,255))#draw in red
    cv2.putText(img, file+" "+str(temperature), (50,100), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 4)
    out.write(img)
out.release()
##########################

###Analysis###############
book = xlwt.Workbook(encoding="utf-8")
sheet = book.add_sheet("Temperatures")
sheet.write(0,0, "Radius (pixels)")
sheet.write(0,1, "freezing temperature")
for row,drop in enumerate(droplets,1):
    if drop.freeze_temperature:
        sheet.write(row,0,drop.r)
        sheet.write(row,1,drop.freeze_temperature)
    else:
        sheet.write(row,0,"Freeze not detected")
book.save(files[0][:-len(file_extension)]+".xls")
##########################


