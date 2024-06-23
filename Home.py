import tkinter as tk
#from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk



##############################################+=============================================================
root = tk.Tk()
root.configure(background="brown")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Alzheimer Disease Detection")

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('A5.jpg')
image2 = image2.resize((w,h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)

# Title Label
label_l1 = tk.Label(root, text="Alzheimer Disease Detection",font=("Times New Roman", 30, 'bold'),
                    background="sky blue", fg="white", width=60, height=2)
label_l1.place(x=0, y=0)

img = Image.open('logo.jpg')
img = img.resize((100,70), Image.ANTIALIAS)
logo_image = ImageTk.PhotoImage(img)

logo_label = tk.Label(root, image=logo_image)
logo_label.image = logo_image
logo_label.place(x=40, y=10)


img=ImageTk.PhotoImage(Image.open("Untitled.jpeg")) #j

img2=ImageTk.PhotoImage(Image.open("braiin.png")) #s1

img3=ImageTk.PhotoImage(Image.open("LP-fotor.png")) #s2


logo_label=tk.Label()
logo_label.place(x=0,y=90)


logo_label1=tk.Label(text=' Alzheimers Disease Detection \n "Thought Those With  Alzheimer Might Forget Us, We  Society Must Remember Them"',compound='bottom',font=("Times New Roman", 17, 'bold', 'italic'),width=100, bg="white", fg="black")
logo_label1.place(x=0,y=550)


#image slidshow
# using recursion to slide to next image
x = 1

# function to change to next image
def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img,width=1800,height=700)
	elif x == 2:
		logo_label.config(image=img2,width=1500,height=500)
	elif x == 3:
		logo_label.config(image=img3,width=1300,height=500)
	x = x+1
	root.after(2000, move)

# calling the function
move()

#Positioning the Frame:
frame_alpr = tk.LabelFrame(root, text=" --Login & Register-- ", width=600, height=100, bd=5, font=('times', 14, ' bold '),bg="white")
frame_alpr.grid(row=0, column=0, sticky='nw') #grid geometry manager label positioning
frame_alpr.place(x=350, y=300) #position of the frame


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#registration function   
def reg():
    from subprocess import call
    call(["python","register.py"])
    root.destroy()
    
#login function 
def log():
    from subprocess import call
    call(["python","Alzheimer_login.py"])
    root.destroy()
    
#exit function 
def window():
  root.destroy()
  
  


button4 = tk.Button(frame_alpr, text="Login", command=log, width=12, height=1,font=('times 15 bold underline'),bd=0,bg="sky blue", fg="white")
button4.place(x=80, y=15)

button4 = tk.Button(frame_alpr, text="Registration", command=reg, width=12, height=1,font=('times 15 bold underline'),bd=0,bg="sky blue", fg="white")
button4.place(x=350, y=15)

root.mainloop()