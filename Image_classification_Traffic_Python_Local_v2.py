#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the required libraries
import keras
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot  as plt
from PIL import Image
from IPython.display import display


# In[2]:


#Set the image size to reresize all the images into (im_size x im_size x 3)
global im_size
im_size = 50 
training_data = []


# In[3]:


#Get the path to the training data
Path=os.getcwd() #Current path

Path=Path+"/Data/Training/" #Folder of the training data
Path=Path.replace('\\', '/')

print (Path)
Classes = next(os.walk(Path))[1] #The name of the folders
print(f"\n{Classes}") #Print the classes 


# In[4]:


def image_processing(full_path = "path"):

    im = cv2.imread(full_path)
    im = cv2.resize(im,(im_size, im_size))
    im = im/255.0
    
    return im


# In[5]:


#Read the training data & label them based on folder name
global label_data
training_data = []
label_data = []

for inx,ima in enumerate(Classes):
    files = next(os.walk(Path+ima))[2]
    
    for file in files:
        this_image = Path+ima+"/"+file
        print("This image",this_image,"\n")
        
        try: 
            im = image_processing(full_path = this_image) #Process this image to store the data in an array
            training_data.append([im, inx]) #Store the array and the index(class) in training_data
            label_data.append([inx,ima]) #Store the index and folder name(class) in label_data
            
        except Exception as e:
            print(f"\n\n\n Error processing this image:{this_image}\n\n\n")
            pass


print("\n\n The Shape of the image is: ",training_data[0][0].shape)

plt.imshow(training_data[1][0])
plt.show()


# In[6]:


#Read the testing data & label them based on folder name

Path=os.getcwd()
Path=Path+"/Data/Validation/"
Path=Path.replace('\\', '/')
print (Path)

testing_data = []
label_data2 = []
for inx,ima in enumerate(Classes):
    files=next(os.walk(Path+ima))[2]
    
    for file in files:
        
        this_image = Path+ima+"/"+file
        
        print("This image",this_image)
        try: 
            
            im = image_processing(full_path = this_image) #Process this image to store the data in an array
            testing_data.append([im, inx]) #Add the array data and the index in testing_data
            label_data2.append([inx,ima]) #Add the index and the folder name(class) in the label_data2
            
            
        except Exception as e:
            print(f"\n\n\n Error processing this image:{this_image}\n\n\n")

            pass

        
print("\n\n The Shape of the image is: ",testing_data[0][0].shape)

plt.imshow(testing_data[1][0])
plt.show()


# In[7]:


#Get the labels names
label_data = pd.DataFrame(label_data) #Convert the label_data in a DataFrame
label_data = label_data[1].unique() #Store the unique values in label_data (each class z)
print(label_data)


# In[8]:


#Shuffle The training and testing data
import random
random.shuffle(training_data)
random.shuffle(testing_data)


# In[9]:


#Seprate the features and labels in the training data  from [Features, label] => X = features, Y = Label

X_train = []
y_train = []
for features,label in training_data:
    X_train.append(features)
    y_train.append(label)
print("Number of training images:",len(y_train),"\nExamples of the labels",y_train[:5])


# In[10]:


#Seprate the features and labels in the training data  from [Features, label] => X = features, Y = Label

X_test = []
y_test = []

for features,label in testing_data:
    X_test.append(features)
    y_test.append(label)

print("Number of testing images:",len(y_test),"\nExamples of the labels",y_test[:5])


# In[11]:


#Reshape the features to (n x im_size x im_size x 3)

X_train = np.array(X_train).reshape(-1, im_size, im_size,3)
X_test = np.array(X_test).reshape(-1, im_size, im_size,3)


# # Code

# In[12]:


print("The shape of the training features is:",X_train.shape)
print("The shape of the testing features is:",X_test.shape)


# In[13]:


#Show an example from the training images
idx = 80
ex = X_train[idx].copy()
plt.imshow(ex)
plt.show()



# Split the image into its BGR channels
blue_channel, green_channel, red_channel = cv2.split(ex)

# Stack the channels into a single NumPy array
ex = np.stack((red_channel, green_channel, blue_channel), axis=-1)


plt.imshow(ex)
plt.show()


# In[15]:


#Impor the libraries for the NN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D
from keras.utils import np_utils



#Build the NN Architechture

model=Sequential()
model.add(Conv2D(128,(3,3), strides=1, input_shape=[im_size, im_size, 3], activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(AveragePooling2D(pool_size=(3,3)))


model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(AveragePooling2D(pool_size=(3,3)))


# model.add(Conv2D(32,(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(AveragePooling2D(pool_size=(3,3)))


#Flat the extracted data
model.add(Flatten())

#Fully Connected NN
model.add(Dense(200,activation='relu'))
# model.add(Dropout(0.1))

model.add(Dense(200,activation='relu'))

model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(len(label_data), activation='softmax')) #softmax, relu, sigmoid, linear


# In[17]:


model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #Adam, SGD


# In[18]:


model.summary()


# In[19]:


#Prepare the training data
X_train = np.asarray(X_train).astype('float32').reshape((-1,im_size,im_size,3))
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))

#Prepare the testing data
X_test = np.asarray(X_test).astype('float32').reshape((-1,im_size,im_size,3))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))


# In[20]:


file_path= "model3.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                               monitor='val_accuracy', 
                                                               save_best_only=True)


history = model.fit(X_train, y_train, epochs=50,
                    callbacks=[model_checkpoint_callback], verbose=1, validation_data=(X_test, y_test))
# history = model.fit(train_generator, epochs=25,validation_data = validation_generator, verbose = 1)


# In[21]:


import matplotlib.pyplot as plt

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


# In[22]:


from keras.models import load_model
model = load_model('model3.h5')


_,acc=model.evaluate(X_test,y_test)
print(acc*100)




import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2
import numpy as np
import pandas as pd

from keras.models import load_model
model = load_model('model3.h5')


def image_display(full_path = "path"):
    image = Image.open(full_path)
    image.thumbnail((400, 400))  # Adjust the size of the thumbnail
    display(image)

def image_processing(full_path = "path"):
    image_display(full_path = full_path)
    im = cv2.imread(full_path)
    im = cv2.resize(im,(im_size, im_size))
    im = im/255.0
        
    return im

def model_pred(im_array):
    
    #im_array = np.array(im_array).reshape(-1, im_size, im_size,3)
    im_array = np.array(im_array).reshape(1, im_size, im_size,3)

    
    pred=model.predict(im_array)
    #print(pred)
    pred = np.array(pred)
    pred = pred.reshape(-1,)
    idx = np.argmax(pred)
    print("\n-----------------------------------------------------\n")
    print("The index of the predicted class:",idx)
    class_ = label_data[idx]

    #The probabilty of other classes
    print("Predicted class: ",class_)
    
    # Calculate the sum of the values
    total = np.sum(pred)
    # Convert the values to percentages
    pred = (pred / total) * 100
    pred = pred.astype("int32")
    print("Probability of other classes: ", pred)

    display_data = pd.DataFrame([pred])
    display_data.columns = label_data

    display(display_data)
    
    return class_
    
    
# label_data = ['Left', 'Pedestrian', 'Right', 'Roundabout', 'Speed 100', 'Speed 120', 'Speed 60', 'Speed 80', 'Stop', 'Traffic light']
# im_size =100


# In[24]:


# Process the image
image_array = image_processing("Data/Validation/Roundabout/Roundabout_4.jpg")

# Use the image array for model prediction
prediction = model_pred(image_array)



# In[31]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2
import numpy as np
import pandas as pd

from keras.models import load_model
label_data = ['Left', 'Pedestrian', 'Right', 'Roundabout', 'Speed 100', 'Speed 120', 'Speed 60', 'Speed 80', 'Stop', 'Traffic light']

model = load_model('model3.h5')

import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np


# Function to handle button click event
def browse_image():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    # Check if a file was selected
    if file_path:
        
        
        # Process the image
        image_array = image_processing(file_path)
        
        # Use the image array for model prediction
        prediction = model_pred(image_array)
        
        # Display the prediction
        prediction_label.configure(text="Prediction: " + prediction)

# Create the main window
window = tk.Tk()
window.title("Image Prediction")

# Create a button to browse for an image
browse_button = tk.Button(window, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Create a label for displaying the prediction
prediction_label = tk.Label(window, text="Prediction: ")
prediction_label.pack(pady=10)

# Create a label for displaying the image (optional)
# image_label = tk.Label(window)
# image_label.pack(pady=10)

# Run the main window loop
window.mainloop()



import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import pandas as pd
import os
import cv2
from keras.models import load_model

im_size=50
# label_data = ['Left', 'Pedestrian', 'Right', 'Roundabout', 'Speed 100', 'Speed 120', 'Speed 60', 'Speed 80', 'Stop', 'Traffic light']
# im_size =100

#load the trained model to classify the images
model = load_model('model3.h5')


#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Image Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):  
    global pred
    
    im = cv2.imread(file_path)
    im = cv2.resize(im,(im_size, im_size))
    #im = np.asarray(im).astype('float32').reshape((im_size,im_size,3))    
    im=im/255.0
    im = np.array(im).reshape(-1, im_size, im_size,3) 

    pred=model.predict(im)
    #print(pred)
    pred = np.array(pred)
    pred = pred.reshape(-1,)
    idx = np.argmax(pred)
    print("______________________________________________\nThe index of the predicted class:",idx)
    class_ = label_data[idx]
    label.configure(foreground='#011638', text=class_)


    #The probabilty of other classes
    print("Predicted image: ",class_)
    # Calculate the sum of the values
    total = np.sum(pred)

    # Convert the values to percentages
    pred = (pred / total) * 100
    pred = pred.astype("int32")
    print("Probability of other classes: ", pred)

    display_data = pd.DataFrame([pred])
    display_data.columns = label_data
    display(display_data)
    


def proba_():
    """Display a button that shows the predicted probabilities."""
    classify_b = Button(
        top,
        text="Probability",
        command=lambda: classify(file_path),
        padx=10,
        pady=5,
    )
    classify_b.configure(
        background="#364156",
        foreground="white",
        font=("arial", 20, "bold"),
    )
    classify_b.place(relx=0.79, rely=0.46)
    
    
    
    
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",
   command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

    
    
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Choose an image",command=upload_image,
  padx=10,pady=5)

upload.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Image Classification",pady=15, font=('arial',20,'bold'))

heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()



top.mainloop()


#['Left', 'Pedestrian', 'Right', 'Roundabout', 'Speed 100', 'Speed 120', 'Speed 60', 'Speed 80', 'Stop', 'Traffic light']

############################################
############################################
############################################
############################################