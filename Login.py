#Imports:
import tkinter as tk
from tkinter import ttk, LEFT, END
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re


#Window Setup:
root = tk.Tk()
root.configure(background="white")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Alzheimer Disease Detection")


username = tk.StringVar()
password = tk.StringVar()
        

#For background Image
image2 = Image.open('1.jpg')
image2 = image2.resize((w,h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)

#Top Frame
label_l1 = tk.Label(root, text="Alzheimer Disease Detection",font=("Times New Roman", 30, 'bold'),
                    background="sky blue", fg="white", width=60, height=2)
label_l1.place(x=0, y=0)

img = Image.open('logo.jpg')
img = img.resize((100,70), Image.ANTIALIAS)
logo_image = ImageTk.PhotoImage(img)

logo_label = tk.Label(root, image=logo_image)
logo_label.image = logo_image
logo_label.place(x=40, y=10)


def registration():
    from subprocess import call
    call(["python","register.py"])
    root.destroy()
    
    
#Database Connection (within Login Function):
def login():
        # To Establish Connection

    with sqlite3.connect('evaluation.db') as db:
         c = db.cursor()

        # Checks for user details, if existing.
         db = sqlite3.connect('evaluation.db')
         cursor = db.cursor()
         cursor.execute("CREATE TABLE IF NOT EXISTS registration"
                           "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT,Gender TEXT,age TEXT , password TEXT)")
         db.commit()
         find_entry = ('SELECT * FROM registration WHERE username = ? and password = ?')
         c.execute(find_entry, [(username.get()), (password.get())])
         result = c.fetchall()

         if result:
            msg = ""
            print(msg)
            ms.showinfo("message", "Logged in Sucessfully")
            # ===========================================
            root.destroy()

            from subprocess import call
            call(['python','GUI_Master_old.py'])

            # ================================================
         else:
           ms.showerror('Oops!', 'Username Or Password Did Not Found/Match.')


bg1_icon=ImageTk.PhotoImage(file="m.jpg")
bg_icon=ImageTk.PhotoImage(file="g1.png")
user_icon=ImageTk.PhotoImage(file="prof.png")
pass_icon=ImageTk.PhotoImage(file="g6.png")
        

title=tk.Label(root, text="Login Here", font=("Times new roman", 25, "bold","italic"),bd=5,bg="silver",fg="white")
title.place(x=550,y=170,width=260)
        
Login_frame=tk.Frame(root,bg="white")
Login_frame.place(x=400,y=250)
        
logolbl=tk.Label(Login_frame,image=bg_icon,bd=0).grid(row=0,columnspan=2,pady=20)
        
lbluser=tk.Label(Login_frame,text="Username",image=user_icon,compound=LEFT,font=("Times new roman", 20, "bold"),bg="white").grid(row=1,column=0,padx=20,pady=10)
txtuser=tk.Entry(Login_frame,bd=5,textvariable=username,font=("",15))
txtuser.grid(row=1,column=1,padx=20)
        
lblpass=tk.Label(Login_frame,text="Password",image=pass_icon,compound=LEFT,font=("Times new roman", 20, "bold"),bg="white").grid(row=2,column=0,padx=50,pady=10)
txtpass=tk.Entry(Login_frame,bd=5,textvariable=password,show="*",font=("",15))
txtpass.grid(row=2,column=1,padx=20)
        
btn_log=tk.Button(Login_frame,text="Login",command=login,width=15,font=("Times new roman", 14, "bold"),bg="light blue",fg="black")
btn_log.grid(row=3,column=1,pady=10)
btn_reg=tk.Button(Login_frame,text="Create Account",command=registration,width=15,font=("Times new roman", 14, "bold"),bg="grey",fg="white")
btn_reg.grid(row=3,column=0,pady=10)
        

# Login Function

def log():
    from subprocess import call
    call(["python","GUI_main.py"])
    root.destroy()
    
def window():
  root.destroy()
  
   
button1 = tk.Button(label_l1, text="Home", command=log, width=12, height=1,font=('times 20 bold underline'),bd=0, bg="sky blue", fg="white")
button1.place(x=1100, y=20)



root.mainloop()