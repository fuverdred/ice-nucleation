import os
import re
import datetime

import cv2
import matplotlib.pyplot as plt
import xlwt#excel

class Droplet:
    def __init__(self, i, j, r,img):
        self.i = int(i)
        self.j = int(j)
        self.r = int(r)
        self.area = r*r#Don't care about pi, just for normalising intensity change with area
        self.intensity = []#normalised to area
        self.freeze_temperature = None#Becomes a number if droplet freezes
        self.find_pixels()
        self.find_intensity(img)

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
        self.intensity.append(sum([img[j][i] for i,j in self.pixels])/self.area)

    def check_frozen(self, temperature, img, threshold = 15):
        #Compare previous intensity to current
        self.find_intensity(img)
        if self.intensity[-1] - self.intensity[-2] > threshold and not self.freeze_temperature:
            self.freeze_temperature = temperature

    def draw(self, img, colour = (0,0,255), thickness = 0.5):
        cv2.circle(img,(self.i,self.j), self.r, colour, thickness)

def check_border(x,y,r,img, border = 10):
    """Check if droplet is hanging over the edge of the picture. input
image must be single channel."""
    Y,X = img.shape
    if x-r < border or \
       x+r > X-border or \
       y-r < border or\
       y+r > Y-border:
        return False
    return True

def stabilise(img,x,y):
    """Takes in integer values for the image displacement. Moves
the image in the opposite direction and adds a border for the cropped
bit."""
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
    """Return a list of the freezing temperature of each droplet, along with the radius of
the droplet"""
    return [[d.r, startT+((d.freeze_frame-start_frame)*DTdn)] for d in droplets if d.frozen]

def main():
    ##CONSTANTS AND PARAMETERS#####################
    file_extension = ".tiff"
    border = 50#Number of pixels to ignore around the edge of the image
    threshold = 12#change in brightness signifying freezing (arbitrary units)
    frameRate = 5#for saved video

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

    circle_params = dict(dp = 1.5,#parameters for detecting droplets. See cv2.HoughCircle docs
                         minDist = 50,
                         param2 = 50,
                         minRadius = 10,
                         maxRadius = 100)
    ###############################################

    plt.ion()#interactive plotting

    ##Sort files##############
    os.chdir("c:/Users/fc12293/ice_pics/Photoresist")
    files = [i for i in os.listdir() if i[-len(file_extension):] == file_extension]
    files.sort(key = alphanum_key)#Human sort of numbers 1->2->3 instead of 1->11->12...
    ##########################

    ##Setting up droplets#####
    img = cv2.imread(files[1])
    img = cv2.medianBlur(img, 5)
    #g = img[:,:,1]#Take all the green pixels
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, **circle_params)[0]
    p1 = cv2.goodFeaturesToTrack(grey, mask = None, **feature_params)#For droplet tracking
    start_frame = int(regex.findall(files[0])[-1])#for working out temperature

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
        #g2 = img[:,:,1]
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

if __name__ == "__main__":
    main()
