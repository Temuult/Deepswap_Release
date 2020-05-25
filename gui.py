import tkinter as tk
from tkinter import ttk,filedialog
import PIL
from PIL import Image, ImageTk

from FaceDet import *
from MdlTrain import *
from ApplyMdl import *


##Start Window
win = tk.Tk()
win.title("Python GUI")
win.geometry("1280x720")
win.resizable(0,0)


##Button action
## TAB1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def b3click():
    textbox3.delete(0, 'end')
    fname=filedialog.askopenfilename(title="Select a file")
    textbox3.insert(0,fname)

def b4click():
    textbox4.delete(0, 'end')
    fname=filedialog.askdirectory(title="Select a directory")
    textbox4.insert(0,fname)
    
def b5click():
    extract(textbox3.get(),textbox4.get()+r'/temp',textboxevr.get())
    generate(textbox4.get())
    clean(textbox4.get()+r'/temp')
    #filt(textbox4.get())
    fn=getfiles(textbox4.get())
    picI=len(fn)/15
    #print(picI)
    for w in range(15):
        print(picI*w)
        pics[w]=textbox4.get()+"/face"+str(picI)+".jpg"
        imgl = Image.open(fn[int(w*picI)])
        imgl = imgl.resize((128, 128), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(imgl)
        pics[w]=imgtk
        panel_list[w].configure(image=imgtk)


        
## TAB2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def bclick():
    textbox.delete(0, 'end')
    fname=filedialog.askdirectory(title="Select a directory")
    textbox.insert(0,fname)
    

def b1click():
    textbox1.delete(0, 'end')
    fname=filedialog.askdirectory(title="Select a directory")
    textbox1.insert(0,fname)

def b6click():
    textbox2.delete(0, 'end')
    fname=filedialog.askdirectory(title="Select a directory")
    textbox2.insert(0,fname)

def b2click():
    labelst.configure(text="Status:"+"")
    Train(textbox.get(),textbox1.get(),textbox2.get(),textbox5.get(),textboxsaveget())

## TAB3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def tb3bclick():
    tb3textbox.delete(0, 'end')
    fname=filedialog.askopenfilename(title="Select a video")
    tb3textbox.insert(0,fname)
    

def tb3b1click():
    tb3textbox1.delete(0, 'end')
    fname=filedialog.askdirectory(title="Select a directory")
    tb3textbox1.insert(0,fname)

def tb3b2click():
    tb3textbox2.delete(0, 'end')
    fname=filedialog.askdirectory(title="Select a directory")
    tb3textbox2.insert(0,fname)

def tb3b3click():
    labelst.configure(text="Status:"+"LOL")
    applymodel(tb3textbox.get(),tb3textbox1.get(),tb3textbox2.get())
    

##Tabs
tab_parent=ttk.Notebook(win)

tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)
tab3 = ttk.Frame(tab_parent)
tab4 = ttk.Frame(tab_parent)
tab5 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text='Extract')
tab_parent.add(tab2, text='Train')
tab_parent.add(tab3, text='Apply model')
tab_parent.add(tab4, text='Tools')
tab_parent.add(tab5, text='About')

tab_parent.pack(expand=1, fill='both')
##General~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##Lable frame
lf3 = ttk.LabelFrame(win, text="Logs",width=1250,height=122)
lf3.place(relx=0.01,rely=0.80)

lf3 = ttk.LabelFrame(win, text="Preview",width=775,height=540)
lf3.place(relx=0.38,rely=0.04)

labelst=tk.Label(win, text="Status:")
labelst.pack()
labelst.place(relx=0.5,rely=0.97)

##TAB1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##Lable frame muthadkr
lf2 = ttk.LabelFrame(tab1, text="Picture and face extraction section options",width=450,height=540)
lf2.place(relx=0.01,rely=0.01)



##Text box
boxtxt3=tk.StringVar()
textbox3=ttk.Entry(lf2, width=45, textvariable=boxtxt3)
textbox3.place(relx=0.27,rely=0.045)
textbox3.insert(0, r"C:/Users/Q/Documents/Deepswap/o.mp4")


boxtxt4=tk.StringVar()
textbox4=ttk.Entry(lf2, width=45, textvariable=boxtxt4)
textbox4.place(relx=0.27,rely=0.1)
textbox4.insert(0, r"C:\Users\Q\Documents\Deepswap\TaOut")\

boxtxtevr=tk.StringVar()
textboxevr=ttk.Entry(lf2, width=5, textvariable=boxtxtevr)
textboxevr.place(relx=0.27,rely=0.155)
textboxevr.insert(0, "1")

##Buttons
button3=tk.Button(lf2, text='O', command=b3click,height = 1, width = 2)
button3.place(relx=0.92,rely=0.0425)

button4=tk.Button(lf2, text='O', command=b4click,height = 1, width = 2)
button4.place(relx=0.92,rely=0.0975)


button5=tk.Button(lf2, text='Extract', command=b5click,height = 1, width = 5)
button5.place(relx=0.85,rely=0.935)

##Labels
label=tk.Label(lf2, text="Input File:")
label.pack()
label.place(relx=0.05,rely=0.045)

label4=tk.Label(lf2, text="Output directory:")
label4.pack()
label4.place(relx=0.05,rely=0.1)

label5=tk.Label(lf2, text="Every x-th frame:")
label5.pack()
label5.place(relx=0.05,rely=0.155)


##Images
##image = Image.open("a.jpg")
##image = image.resize((128, 128), Image.ANTIALIAS)
##img = ImageTk.PhotoImage(image)
##panel = tk.Label(lf3, image = img,)
##panel.pack()
##panel.place(relx=0.01,rely=0.01)
ph="ph.jpg"
pics = [ph for i in range(15)]
panel_list = []
wx=0
wy=0
for w in range(15):
    ##pics[w]="img"+str(w)+".jpg"
    imgl = Image.open(pics[w])
    imgl = imgl.resize((128, 128), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(imgl)
    pics[w]=imgtk
    panel_list.append(tk.Label(lf3, image = imgtk))
    if(w%5==0 and w!=0):
        wy+=0.325
        wx=0
    panel_list[w].place(relx=0.01+wx,rely=0.045+wy)
    wx+=0.2

##TAB2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##Lable frame muthadkr
lf1 = ttk.LabelFrame(tab2, text="Model training options",width=450,height=540)
lf1.place(relx=0.01,rely=0.01)

##Text box
boxtxt=tk.StringVar()
textbox=ttk.Entry(lf1, width=45, textvariable=boxtxt)
textbox.place(relx=0.27,rely=0.045)
textbox.insert(0, r"C:\Users\Q\Documents\Deepswap\Oout")

boxtxt1=tk.StringVar()
textbox1=ttk.Entry(lf1, width=45, textvariable=boxtxt1)
textbox1.place(relx=0.27,rely=0.1)
textbox1.insert(0, r"C:\Users\Q\Documents\Deepswap\TaOut")

boxtxt2=tk.StringVar()
textbox2=ttk.Entry(lf1, width=45, textvariable=boxtxt2)
textbox2.place(relx=0.27,rely=0.155)
textbox2.insert(0, r"C:\Users\Q\Documents\Deepswap\nwmdl")

boxtxt5=tk.StringVar()
textbox5=ttk.Entry(lf1, width=45, textvariable=boxtxt5)
textbox5.place(relx=0.27,rely=0.21)
textbox5.insert(0, r"120")

boxtxtsave=tk.StringVar()
textboxsave=ttk.Entry(lf1, width=45, textvariable=boxtxtsave)
textboxsave.place(relx=0.27,rely=0.265)
textboxsave.insert(0, r"100")


##Buttons
button=tk.Button(lf1, text='O', command=bclick,height = 1, width = 2)
button.place(relx=0.92,rely=0.04)

button1=tk.Button(lf1, text='O', command=b1click,height = 1, width = 2)
button1.place(relx=0.92,rely=0.095)

button6=tk.Button(lf1, text='O', command=b6click,height = 1, width = 2)
button6.place(relx=0.92,rely=0.15)

##button7=tk.Button(lf1, text='O', command=b6click,height = 1, width = 2)
##button7.place(relx=0.92,rely=1.425)

button2=tk.Button(lf1, text='Train', command=b2click,height = 1, width = 5)
button2.place(relx=0.85,rely=0.935)

##Labels
label1=tk.Label(lf1, text="Original Faces:")
label1.pack()
label1.place(relx=0.05,rely=0.045)

label2=tk.Label(lf1, text="Target Faces:")
label2.pack()
label2.place(relx=0.05,rely=0.1)

label3=tk.Label(lf1, text="Model:")
label3.pack()
label3.place(relx=0.05,rely=0.155)

label6=tk.Label(lf1, text="Epochs:")
label6.pack()
label6.place(relx=0.05,rely=0.21)

label6=tk.Label(lf1, text="Checkpoint:")
label6.pack()
label6.place(relx=0.05,rely=0.265)

##TAB3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##Lable frame muthadkr
tb3_lf1 = ttk.LabelFrame(tab3, text="Applying model:",width=450,height=540)
tb3_lf1.place(relx=0.01,rely=0.01)

##Text boxes
tb3boxtxt=tk.StringVar()
tb3textbox=ttk.Entry(tb3_lf1, width=45, textvariable=tb3boxtxt)
tb3textbox.place(relx=0.27,rely=0.045)
tb3textbox.insert(0, r"C:/Users/Q/Documents/Deepswap/bshrt2.mp4")


tb3boxtxt1=tk.StringVar()
tb3textbox1=ttk.Entry(tb3_lf1, width=45, textvariable=tb3boxtxt1)
tb3textbox1.place(relx=0.27,rely=0.1)
tb3textbox1.insert(0, r"C:/Users/Q/Documents/Deepswap/vidout")

tb3boxtxt2=tk.StringVar()
tb3textbox2=ttk.Entry(tb3_lf1, width=45, textvariable=tb3boxtxt2)
tb3textbox2.place(relx=0.27,rely=0.155)
tb3textbox2.insert(0, r"C:/Users/Q/Documents/Deepswap/nwmdl")

tb3boxtxt3=tk.StringVar()
tb3textbox3=ttk.Entry(tb3_lf1, width=45, textvariable=tb3boxtxt3)
tb3textbox3.place(relx=0.27,rely=0.21)
tb3textbox3.insert(0, r"120")

##Buttons
tb3button=tk.Button(tb3_lf1, text='O', command=tb3bclick,height = 1, width = 2)
tb3button.place(relx=0.92,rely=0.04)

tb3button1=tk.Button(tb3_lf1, text='O', command=tb3b1click,height = 1, width = 2)
tb3button1.place(relx=0.92,rely=0.095)

tb3button2=tk.Button(tb3_lf1, text='O', command=tb3b2click,height = 1, width = 2)
tb3button2.place(relx=0.92,rely=0.15)

tb3button3=tk.Button(tb3_lf1, text='Apply', command=tb3b3click,height = 1, width = 5)
tb3button3.place(relx=0.85,rely=0.935)

##Labels
tb3label=tk.Label(tb3_lf1, text="Video file:")
tb3label.pack()
tb3label.place(relx=0.05,rely=0.045)

tb3label1=tk.Label(tb3_lf1, text="Output path:")
tb3label1.pack()
tb3label1.place(relx=0.05,rely=0.1)
6
tb3label2=tk.Label(tb3_lf1, text="Models path:")
tb3label2.pack()
tb3label2.place(relx=0.05,rely=0.155)

tb3label3=tk.Label(tb3_lf1, text="Epochs:")
tb3label3.pack()
tb3label3.place(relx=0.05,rely=0.21)

##TAB5~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
labelm=tk.Label(tab5, text="",font=("Arial",28))
labelm.pack()
labelm.place(relx=0,rely=0.5)

##Main loop
win.mainloop()


##TRASH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##,font=("Ch ForeignerLight"



##label1.configure(text=fname)
##askopenfilename(initialdir="/c",title="Select a file")

