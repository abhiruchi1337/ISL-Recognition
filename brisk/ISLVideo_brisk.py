import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import pickle
import numpy as np

alph=list("ABCDEFGHIKLMNOPQRSTUVWXY")
alphabetmap=dict()
for i in range(24):
    alphabetmap[i]=alph[i]
print(alphabetmap)

    
    
def maskAndDetect(frame):
    frame = cv2.resize(frame,(128,128))
    #frame=cv2.flip(frame,1)
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV

    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    cv2.imshow("masked",skinMask)
    
    skinMask = cv2.medianBlur(skinMask, 5)
    
    skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
    
    img2 = cv2.Canny(skin,60,60)
    cv2.imshow("edge detection",img2)
    
    
    brisk = cv2.BRISK_create()
    img2 = cv2.resize(img2,(256,256))
    kp, des = brisk.detectAndCompute(img2,None)
    img2 = cv2.drawKeypoints(img2,kp,None)
    cv2.imshow('Surf detection',img2)
    return des


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        self.textlabel=tk.Label(window, width=40 ,font="Helvetica 18  bold ",bd=4,
		 text="Indian Sign Language Character Recognition ",fg="dark blue").grid( row=1,column=1,columnspan=5)
        self.canvas = tk.Canvas(window,  highlightthickness=5, relief="ridge", width = self.vid.width,height = self.vid.height,border=0.5)
        self.canvas.grid(row=2,column=1,columnspan=5,rowspan=6)
        self.output = tk.StringVar()
        self.output.set('empty')
        self.textlabel=tk.Label().grid()
        self.textlabel=tk.Label(window, width=20 ,relief="groove",font="Helvetica 18  bold ",bd=4,
		 text="Character Detected : ",fg="dark blue").grid(row=9,column=1)
      

        self.textlabel=tk.Label(window, 
		 textvariable=self.output,
		 fg = "dark blue",
		 font = "Helvetica 18 bold ", width=25,relief="groove",bd=4).grid(row=9,column=3)
        self.textlabel=tk.Label().grid( )
        self.delay = 15
        self.svc=pickle.load(open('svm_trained_brisk.sav','rb'))
        self.cluster_model=pickle.load(open('cluster_model_new.sav','rb'))
        self.counter=0
        self.ops=[]
        self.update()
        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),1)))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            frame=cv2.flip(frame,1)
            imgdesc=[maskAndDetect(frame)]
            try:
                img_clustered_words = [self.cluster_model.predict(raw_words) for raw_words in imgdesc]
                img_bow_hist = np.array([np.bincount(clustered_words, minlength=150) for clustered_words in img_clustered_words])

                X = img_bow_hist
                result=self.svc.predict(X)
                
                print(result)
                self.counter+=1
                self.updateOutput(result)
            except:
                self.output.set("No alphabet detected")
                print('oops!, empty frame')
        self.window.after(self.delay, self.update)
    def updateOutput(self, res):
        if self.counter<5:
            self.ops.append(res[0])
        else:
            self.counter=0
            result=max(self.ops,key=self.ops.count)
            self.ops=[]
            self.output.set(alphabetmap[result])


class MyVideoCapture:
    def __init__(self, height, video_source=0):
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, frame)
            else:
                return (ret, None)
        else:
            return (ret, None)
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        
App(tk.Tk(), "Sign Language recognition")
