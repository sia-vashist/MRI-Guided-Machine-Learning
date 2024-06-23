#Imports:
import tkinter as tk
#from tkinter import ttk, LEFT, END
from tkinter import messagebox as ms # to show pop up message
import sqlite3
from PIL import Image, ImageTk
import re #regular expression library for email validation

#===============================================================================
#Window Setup:
root = tk.Tk()
root.configure(background="white")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Alzheimer Disease Detection")


# ++++++++++++++++++++++++++++++++++++++++++++
#For background Image
image2 = Image.open('brainstim.jpg')
image2 = image2.resize((w,h), Image.LANCZOS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)



#Header Label:
label_l1 = tk.Label(root, text="Alzheimer Disease Detection",font=("Times New Roman", 30, 'bold'),
                    background="#133C55", fg="white", width=67, height=2)
label_l1.place(x=-90, y=0)

#Logo Image:
img = Image.open('logo.jpg')
img = img.resize((100,70), Image.LANCZOS)
logo_image = ImageTk.PhotoImage(img)

logo_label = tk.Label(root, image=logo_image)
logo_label.image = logo_image
logo_label.place(x=40, y=10)

#Registration Frame:
frame_alpr = tk.LabelFrame(root, text=" --Register-- ", width=600, height=550, bd=5, font=('times', 14, ' bold '),fg="white",bg="steel blue")
frame_alpr.place(x=10, y=97)

#Registration Form Frame:
Fullname = tk.StringVar()
address = tk.StringVar()
username = tk.StringVar()
Email = tk.StringVar()
Phoneno = tk.IntVar()
password = tk.StringVar()
password1 = tk.StringVar()


#Database Connection:
# database code
db = sqlite3.connect('evaluation.db')
cursor = db.cursor() #sql queries
cursor.execute("CREATE TABLE IF NOT EXISTS registration"
               "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT, password TEXT)")
db.commit()

#Password Check Function:
def password_check(passwd): 
	
	SpecialSym =['$', '@', '#', '%'] 
	val = True
	
	if len(passwd) < 6: 
		print('length should be at least 6') 
		val = False
		
	if len(passwd) > 20: 
		print('length should be not be greater than 8') 
		val = False
		
	if not any(char.isdigit() for char in passwd): 
		print('Password should have at least one numeral') 
		val = False
		
	if not any(char.isupper() for char in passwd): 
		print('Password should have at least one uppercase letter') 
		val = False
		
	if not any(char.islower() for char in passwd): 
		print('Password should have at least one lowercase letter') 
		val = False
		
	if not any(char in SpecialSym for char in passwd): 
		print('Password should have at least one of the symbols $@#') 
		val = False
	if val: 
		return val 

#Registration Function:
def insert():
    fname = Fullname.get()
    addr = address.get()
    un = username.get()
    email = Email.get()
    mobile = Phoneno.get()
    pwd = password.get()
    cnpwd = password1.get()

    with sqlite3.connect('evaluation.db') as db:
        c = db.cursor()

    # Find Existing username if any take proper action
    find_user = ('SELECT * FROM registration WHERE username = ?')
    c.execute(find_user, [(username.get())])


    regex='^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    if (re.search(regex, email)):
        a = True
    else:
        a = False
    # validation
    if (fname.isdigit() or (fname == "")):
        ms.showinfo("Message", "please enter valid name")
    elif (addr == ""):
        ms.showinfo("Message", "Please Enter Address")
    elif (email == "") or (a == False):
        ms.showinfo("Message", "Please Enter valid email")
    elif((len(str(mobile)))<10 or len(str((mobile)))>10):
        ms.showinfo("Message", "Please Enter 10 digit mobile number")
    elif (c.fetchall()):
        ms.showerror('Error!', 'Username Taken Try a Diffrent One.')
    elif (pwd == ""):
        ms.showinfo("Message", "Please Enter valid password")
    elif(pwd=="")or(password_check(pwd))!=True:
        ms.showinfo("Message", "password must contain atleast 1 Uppercase letter,1 symbol,1 number")
    elif (pwd != cnpwd):
        ms.showinfo("Message", "Password Confirm password must be same")
    else:
        conn = sqlite3.connect('evaluation.db')
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO registration(Fullname, address, username, Email, Phoneno, password) VALUES(?,?,?,?,?,?)',
                (fname, addr, un, email, mobile, pwd))

            conn.commit()
            db.close()
            ms.showinfo('Success!', 'Account Created Successfully !')
            # window.destroy()
            root.destroy()
            from subprocess import call
            call(["python", "Alzheimer_login.py"])
            

#Registration Form Labels and Entry Fields:
l1 = tk.Label(frame_alpr, text="__Registration Form__", font=("Times new roman", 30, "bold","italic"),bd=5, bg="#003459", fg="white")
l1.place(x=120, y=10)

l2 = tk.Label(frame_alpr, text="Full Name :", width=12, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l2.place(x=100, y=100)
t1 = tk.Entry(frame_alpr, textvar=Fullname, width=20, font=('', 15))
t1.place(x=300, y=100)


l3 = tk.Label(frame_alpr, text="Address :", width=12, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l3.place(x=100, y=150)
t2 = tk.Entry(frame_alpr, textvar=address, width=20, font=('', 15))
t2.place(x=300, y=150)


l5 = tk.Label(frame_alpr, text="E-mail :", width=12, font=("Times new roman", 15, "bold"), bd=5,fg="black")
l5.place(x=100, y=200)
t4 = tk.Entry(frame_alpr, textvar=Email, width=20, font=('', 15))
t4.place(x=300, y=200)

l6 = tk.Label(frame_alpr, text="Phone number :", width=12, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l6.place(x=100, y=250)
t5 = tk.Entry(frame_alpr, textvar=Phoneno, width=20, font=('', 15))
t5.place(x=300, y=250)

l4 = tk.Label(frame_alpr, text="User Name :", width=12, font=("Times new roman", 15, "bold"), bd=5,fg="black")
l4.place(x=100, y=300)
t3 = tk.Entry(frame_alpr, textvar=username, width=20, font=('', 15))
t3.place(x=300, y=300)

l9 = tk.Label(frame_alpr, text="Password :", width=12, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l9.place(x=100, y=350)
t9 = tk.Entry(frame_alpr, textvar=password, width=20, font=('', 15), show="*")
t9.place(x=300, y=350)

l10 = tk.Label(frame_alpr, text="Confirm Password:", width=13, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l10.place(x=100, y=400)

t10 = tk.Entry(frame_alpr, textvar=password1, width=20, font=('', 15), show="*")
t10.place(x=300, y=400)

#Registration Button:
btn = tk.Button(frame_alpr, text="Register", bg="light cyan",font=("times new roman",20,"bold"),fg="black", width=9, height=1, bd=5,command=insert)
btn.place(x=220, y=450)


def log():
    from subprocess import call
    call(["python","Alzheimer_login.py"])
    root.destroy()
    
def window():
  root.destroy()
  

label_l1 = tk.Label(root, text="** 2024 FINAL YEAR PROJECT - DYPIU **",font=("Times New Roman", 10, 'bold'),
                    background="BLACK", fg="white", width=250, height=2)
label_l1.place(x=0, y=800)


root.mainloop()
