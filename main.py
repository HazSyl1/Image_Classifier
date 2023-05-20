import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from tkinter import messagebox
import pickle
import tensorflow
from tensorflow import keras
from keras.models import load_model

#load the trained model to classify the images


model = load_model('CIFAR_10_epochs_10.h5')
#model=pickle.load(open("model_cifar10[acc-87]",'rb'))
#dictionary to label all the CIFAR-10 dataset classes.

classes = {
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
}
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('IMAGE CLASSIFIER')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def on_close():
    if messagebox.askyesno(title="QUIT",message="Do you want to QUIT"):
        top.destroy()

menubar=tk.Menu(top)
filemenu=tk.Menu(menubar,tearoff=0)
filemenu.add_command(label="Close",command=on_close)
menubar.add_cascade(menu=filemenu,label="File")
top.config(menu=menubar)
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])[0]
    result = np.argmax(pred)
    sign=classes[result]
    print(sign)
    label.configure(foreground='#011638', text=sign)
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#519bc9', foreground='white',font=('arial',14,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload=Button(top,text="Upload an Image",command=upload_image,
  padx=10,pady=5)
upload.configure(background='#519bc9', foreground='white',
    font=('arial',15,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="IMAGE CLASIFIER",pady=20, font=('arial',20,'bold'))
heading.configure(background='#519bc9',foreground='white')
heading.pack()
top.configure(background='#8544c7')
top.protocol("WM_DELETE_WINDOW",on_close)
top.mainloop()