import tkinter as tk
from tkinter import ttk, LEFT, END
from tkinter import Canvas
from PIL import Image , ImageTk 
import cv2
from tkinter import filedialog
import pydicom    
import os
import numpy as np
import time
import CNNModel 
import sqlite3
global fn, preprocessed
fn=""
preprocessed = False  # Flag to track if the image has been preprocessed
##############################################+=============================================================

#Root Window Configuration:
root = tk.Tk()
root.configure(background="seashell2")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Alzheimer Disease detection")


#For background Image
image2 =Image.open('ss.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)

#Title Label:
lbl = tk.Label(root, text="Alzheimer Disease detection", font=('times', 35,' bold '), height=1,width=50, bg="#0A1173",fg="white")
lbl.place(x=0, y=0)

#Frame for Process:
frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=350, bd=5, font=('times', 14, ' bold '),bg="sky blue")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=100)


#Updates Result Label

def update_label(str_T):
    # Clear previous label if exists
    for widget in root.winfo_children():
        if isinstance(widget, tk.Label) and widget.winfo_y() == 420:
            widget.destroy()
    
    # Create a new label with adjusted size
    result_label = tk.Label(root, text=str_T, width=90, font=("bold", 15), bg='silver', fg='black', wraplength=700)
    result_label.place(x=300, y=300)  # Adjusted y-coordinate
    
    # Limiting the width and height of the result text box
    result_label.config(wraplength=800)

#Result Label
def update_cal(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='silver', fg='black')
    result_label.place(x=350, y=350)
    

#After Preprocessing - Prediction button
def train_model():
 
    update_label("Model Training Start...............")
    
    start = time.time()

    X= CNNModel.main()
    print(X)
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    update_label(msg)

import functools
import operator


def convert_str_to_tuple(tup):
    s = functools.reduce(operator.add, (tup))
    return s

#Loads Model & Prediction
def test_model_proc(fn):
    from keras.models import load_model
    from tensorflow.keras.optimizers import Adam

    IMAGE_SIZE = 64

    if fn:
        # Load the model
        model = load_model('New1_model.h5')
        
        # Load and preprocess the image
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img)
        
        # Check the number of channels and convert to RGB if necessary
        if img.shape[-1] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        img = img.astype('float32')
        img = img / 255.0
        
        # Make prediction
        prediction = model.predict(img)
        yoga = np.argmax(prediction)
        
        # Evaluate CDR score
        cdr_score = evaluate_cdr(yoga)
        
        # Generate result message
        if yoga == 0:
            Cd = "Mild Alzheimer's Detected\n\n Analysis: Our analysis indicates early signs of Alzheimer's disease.\nRecommendation: We recommend consulting a healthcare professional for further evaluation."
        elif yoga == 1:
            Cd = "Moderate Alzheimer's Detected\n\n Analysis: Our analysis suggests moderate progression of Alzheimer's disease. \nRecommendation: We advise seeking immediate medical attention & exploring available treatment options."
        elif yoga == 2:
            Cd = "No Alzheimer's Detected\n\n Analysis: Our analysis found no evidence of Alzheimer's disease.\nRecommendation: Regular cognitive assessments are recommended to monitor brain health."
        elif yoga == 3:
            Cd = "Severe Alzheimer's Detected\n\n Analysis: Our analysis reveals advanced stages of Alzheimer's disease. \nRecommendation: Urgent medical intervention is advised to manage symptoms & enhance quality of life."
       
        # Includes assigned CDR score in the result message
        result_message = Cd + "\nCDR Score: {}".format(cdr_score)
        
        return result_message


def update_label(str_T):
    # Clear previous label if exists
    for widget in root.winfo_children():
        if isinstance(widget, tk.Label) and widget.winfo_y() == 420:
            widget.destroy()
                   
    result_label = tk.Label(root, text=str_T, width=85, font=("bold", 15), bg='#D3D3D3', fg='black', wraplength=700)
    result_label.config(highlightthickness=5, highlightbackground=root['bg'])  # Making the label transparent
    result_label.place(x=300, y=390)  # Adjusted y-coordinate
    
#Prediction:
def test_model():
    global fn, preprocessed
    if fn!="" and preprocessed:
        update_label("Model Testing Start...............")
        
        start = time.time()
    
        X=test_model_proc(fn)
        
        X1="MRI Detection: {0}".format(X)
        
        end = time.time()
            
        ET="Execution Time: {0:.4} seconds \n".format(end-start)
        
        msg="Image Testing Completed.."+'\n'+ X1 + '\n'+ ET
        fn=""
        preprocessed = False  # Reset preprocessing flag after prediction
    else:
        msg="Please preprocess the image before prediction."
        
    update_label(msg)

def evaluate_cdr(prediction):
    if prediction == 0:
        cdr_score = 1.0  # Mild Alzheimer's
    elif prediction == 1:
        cdr_score = 2.0  # Moderate Alzheimer's
    elif prediction == 2:
        cdr_score = 0.0  # No Alzheimer's
    elif prediction == 3:
        cdr_score = 3.0  # Severe Alzheimer's
    else:
        cdr_score = None  # Invalid prediction
    
    return cdr_score

#=============================================================================
#Image PreProcessing
from tkinter import filedialog, messagebox

def openimage():
    global fn, preprocessed #allowing it to be modified inside the function and accessed from outside.
    fileName = filedialog.askopenfilename(initialdir='Use-input_data_resized', title='Select image for Analysis ',
                                       filetypes=[("Dicom File", "*.dcm"), ("All Files", "*.*")])
    if fileName:
        if not fileName.lower().endswith('.dcm'):
            messagebox.showerror("Input Error", "Wrong Format! Please insert DCM file.")
            return

        IMAGE_SIZE = 200
        #Assigns the selected file path to variables
        imgpath = fileName
        fn = fileName

        # Use pydicom to read DICOM file
        ds = pydicom.dcmread(imgpath)

        #Extracts the pixel data from the DICOM file
        img = ds.pixel_array

        # Normalize pixel values for standarization
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

        # Convert numpy array to PIL Image, Image object from an array-like object
        img = Image.fromarray(img.astype(np.uint8)) #reps 0 to 255

        # Resize image
        img = img.resize((IMAGE_SIZE, 200))

        # Save the converted image as a PNG file
        png_filename = os.path.splitext(os.path.basename(imgpath))[0] + '.png'
        save_path = os.path.join('output_folder', png_filename)  # Change 'output_folder' to your desired folder
        fn = save_path
        img.save(save_path)

        # Display image using Tkinter
        im = ImageTk.PhotoImage(img)
        img_label = tk.Label(root, text='Original', font=('times new roman', 20, 'bold'), image=im,
                             compound='bottom', height=250, width=250)
        img_label.image = im
        img_label.place(x=300, y=100)
        preprocessed = False  # Reset preprocessing flag when a new image is loaded

   # out_label.config(text=imgpath)
   
def convert_grey():
    global fn, preprocessed
    if not fn:
        messagebox.showerror("Error", "No image selected. Please select an image first.")
        return
    IMAGE_SIZE = 200
    
    # Load the image using PIL
    img = Image.open(fn)
    
    # Resize the image
    img = img.resize((IMAGE_SIZE, 200))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Converts the image to RGB format (if it's not already)
    if img_array.ndim == 2: #2 dimensions
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Converts the image to grayscale
    gs = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Displays the grayscale image
    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    img2 = tk.Label(root, text='Gray', font=('times new roman', 20, 'bold'), image=imgtk, compound='bottom', height=250, width=250, bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)

    # Converts to binary image using thresholding
    threshold = 120  # You can adjust this threshold as needed
    binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    binary_image = Image.fromarray(binary_array)
    
    # Displays the binary image using Tkinter
    im_binary = ImageTk.PhotoImage(binary_image)
    img_label_binary = tk.Label(root, text='Binary', font=('times new roman', 20, 'bold'), image=im_binary, compound='bottom', height=250, width=250)
    img_label_binary.image = im_binary  # To prevent garbage collection
    img_label_binary.place(x=880, y=100)
    preprocessed = True  # Set preprocessing flag to True
        
#End
def window():
    root.destroy()


button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="#0A1174",fg="white")
button1.place(x=10, y=50)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="#0A1174",fg="white")
button2.place(x=10, y=120)

button4 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model,width=15, height=1,bg="#0A1174",fg="white", font=('times', 15, ' bold '))
button4.place(x=10, y=190)

exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '),bg="brown",fg="white")
exit.place(x=10, y=260)



root.mainloop()