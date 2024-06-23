#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Setting Up Data Generators for CNN Model

# In[23]:


# Importing necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import os
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend as K
import h5py

# Suppress the specific deprecation warning
import warnings
warnings.filterwarnings("ignore")

# Define basepath
basepath = "D:/100% Alzheimer_Diseases_detection/Alzheimer_Diseases"

# Data generators for CNN model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    basepath + '/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    basepath + '/testing_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Define custom loss function
def custom_loss(y_true, y_pred):
    # Custom loss calculation
    cce = K.categorical_crossentropy(y_true, y_pred)
    class_weight = 2.0
    misclassified_class3 = K.cast(K.equal(K.argmax(y_true, axis=-1), 3), dtype=K.floatx()) * K.cast(K.not_equal(K.argmax(y_pred, axis=-1), 3), dtype=K.floatx())
    weighted_loss = cce + (class_weight * K.mean(cce * misclassified_class3, axis=-1))
    return weighted_loss


# # Building and Compiling the CNN Model

# In[24]:


# Build the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu' , kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compile the CNN model with Adam optimizer
cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss= custom_loss,
                  metrics=['accuracy'])

# Show model summary
cnn_model.summary()


# In[25]:


# Define a learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    decay_factor = 0.5
    decay_epochs = 10
    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))
    return lr

# Train the CNN model with Early Stopping and ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(basepath + '/best_modelL.h5', save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the CNN model
history_cnn = cnn_model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[early_stopping, model_checkpoint, lr_scheduler])


#Saving the model
#import h5py
cnn_model.save(basepath + '/New_model.h5')

# In[26]:


# Plot training and validation accuracy and loss for CNN model
fig_cnn = go.Figure()
fig_cnn.add_trace(go.Scatter(x=np.arange(1, 11), y=history_cnn.history['accuracy'], mode='lines', name='Training Accuracy'))
fig_cnn.add_trace(go.Scatter(x=np.arange(1, 11), y=history_cnn.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
fig_cnn.update_layout(title='Training and Validation Accuracy for CNN Model', xaxis_title='Epoch', yaxis_title='Accuracy')

fig_cnn.show()


# ## Observation: 
# ### Training and Validation Accuracy for CNN Model:
# - The training accuracy generally increases with each epoch, indicating that the model is learning and improving its performance on the training data.
# - The validation accuracy also tends to increase initially, which shows that the model is also performing well on data it hasn't seen during training.
# - If the validation accuracy starts to plateau or decrease while the training accuracy continues to increase, it could indicate overfitting, where the model is memorizing the training data without generalizing well to new data.

# In[27]:


fig_loss_cnn = go.Figure()
fig_loss_cnn.add_trace(go.Scatter(x=np.arange(1, 11), y=history_cnn.history['loss'], mode='lines', name='Training Loss'))
fig_loss_cnn.add_trace(go.Scatter(x=np.arange(1, 11), y=history_cnn.history['val_loss'], mode='lines', name='Validation Loss'))
fig_loss_cnn.update_layout(title='Training and Validation Loss for CNN Model', xaxis_title='Epoch', yaxis_title='Loss')

fig_loss_cnn.show()


# ### Training and Validation Loss for CNN Model:
# - The training loss decreases over epochs, indicating that the model is optimizing its parameters to minimize the error on the training data.
# - The validation loss initially decreases as well, but if it starts to increase while the training loss decreases, it suggests overfitting.
# - If both training and validation loss decrease steadily, it indicates that the model is learning effectively.
# - Large gaps between training and validation loss suggest overfitting, meaning the model is fitting too closely to the training data and not generalizing well to new data.

# In[28]:


# Evaluate the CNN model after training for epochs

# Predict classes using the trained CNN model
cnn_predictions = cnn_model.predict(test_generator)
# Get the predicted classes by taking the index of the highest probability
cnn_predicted_classes = np.argmax(cnn_predictions, axis=1)
# Get the true classes from the test generator
cnn_true_classes = test_generator.classes


# In[29]:


# Calculate evaluation metrics for the CNN model after 50 epochs
# Evaluate the CNN model
CNN_loss, CNN_accuracy = cnn_model.evaluate(test_generator)

# Print CNN model evaluation metrics
print("\nCNN Model Evaluation Metrics:")
print("Loss:", CNN_loss)
print("Accuracy:", CNN_accuracy)


# In[30]:


# Find misclassified images
import numpy as np
import random  
import os
import cv2
import matplotlib.pyplot as plt

# Initialize lists to store misclassified images for each class
misclassified_idx_cnn = np.where(cnn_predicted_classes != cnn_true_classes)[0]
misclassified_images_by_class = [[] for _ in range(4)]

# Iterate through misclassified indices and group images by true class
for idx in misclassified_idx_cnn:
    true_label = cnn_true_classes[idx]
    img_path = test_generator.filepaths[idx]
    misclassified_images_by_class[true_label].append(img_path)

# Show 5 misclassified examples for each class
plt.figure(figsize=(20, 12))
for true_label in range(4):  # Classes 0, 1, 2, 3
    class_images = misclassified_images_by_class[true_label]
    num_examples_to_show = min(len(class_images), 5)
    random_class_images = random.sample(class_images, num_examples_to_show)
    for i, img_path in enumerate(random_class_images):
        predicted_label = list(test_generator.class_indices.keys())[cnn_predicted_classes[test_generator.filepaths.index(img_path)]]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 5, true_label * 5 + i + 1)
        plt.imshow(img)
        plt.title(f'True: {true_label}\nPredicted: {predicted_label}')
        plt.axis('off')
plt.tight_layout()
plt.show()


# In[31]:


from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
conf_matrix = confusion_matrix(cnn_true_classes, cnn_predicted_classes)
import seaborn as sns

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='g', 
            xticklabels=['No AD', 'Mild AD', 'Moderate AD', 'Severe AD'], 
            yticklabels=['No AD', 'Mild AD', 'Moderate AD', 'Severe AD'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[32]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import clone_model
from keras.utils import to_categorical

# Define the custom loss function
def custom_loss(y_true, y_pred):
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    class_weight = 2.0
    misclassified_class3 = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), 3), dtype=tf.float32) * tf.cast(tf.not_equal(tf.argmax(y_pred, axis=-1), 3), dtype=tf.float32)
    weighted_loss = cce + (class_weight * tf.reduce_mean(cce * misclassified_class3, axis=-1))
    return weighted_loss

# Load the pre-trained model with custom loss function
pretrained_model = tf.keras.models.load_model("D:/100% Alzheimer_Diseases_detection/Alzheimer_Diseases/best_modelL.h5",
                                              custom_objects={'custom_loss': custom_loss})

# Identify misclassified images and define training data and labels
training_data = []
training_labels = []

for idx in misclassified_idx_cnn:
    img_path = test_generator.filepaths[idx]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64)) / 255.0  # Preprocess image
    training_data.append(img)
    true_label = cnn_true_classes[idx]
    training_labels.append(true_label)

# Convert training labels to one-hot encoded format
training_labels_one_hot = to_categorical(training_labels, num_classes=4)

# Create a new model for fine-tuning
fine_tuned_model = clone_model(pretrained_model)
fine_tuned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Retrain the model using misclassified images
fine_tuned_model.fit(np.array(training_data), training_labels_one_hot, epochs=10, validation_split=0.2)

# Fine-tune the model with a smaller learning rate
fine_tuned_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
fine_tuned_model.fit(np.array(training_data), training_labels_one_hot, epochs=10, validation_split=0.2)

# Evaluate the model on the misclassified images
evaluation = fine_tuned_model.evaluate(test_generator)
print("Fine-tuned Model Evaluation:", evaluation)


# In[34]:


import os
import cv2
import random

# Initialize lists to store correctly classified images for each class
correctly_classified_images_by_class = [[] for _ in range(4)]

# Iterate through correctly classified indices and group images by true class
correctly_classified_idx_cnn = np.where(cnn_predicted_classes == cnn_true_classes)[0]
for idx in correctly_classified_idx_cnn:
    true_label = cnn_true_classes[idx]
    img_path = test_generator.filepaths[idx]
    correctly_classified_images_by_class[true_label].append(img_path)

# Show 5 correctly classified examples for each class
plt.figure(figsize=(20, 12))
for true_label in range(4):  # Classes 0, 1, 2, 3
    class_images = correctly_classified_images_by_class[true_label]
    num_examples_to_show = min(len(class_images), 5)
    random_class_images = random.sample(class_images, num_examples_to_show)
    for i, img_path in enumerate(random_class_images):
        predicted_label = list(test_generator.class_indices.keys())[cnn_predicted_classes[test_generator.filepaths.index(img_path)]]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 5, true_label * 5 + i + 1)
        plt.imshow(img)
        plt.title(f'True/Predicted: {true_label}/{predicted_label}')
        plt.axis('off')
plt.tight_layout()
plt.show()


# # Model Evaluation

# In[35]:


# Load and preprocess data
X_train = []
y_train = []
X_test = []
y_test = []

# Load and preprocess training data
for images, labels in train_generator:
    X_train.append(images.reshape(images.shape[0], -1))
    y_train.append(labels.argmax(axis=1))
    if len(X_train) * 32 >= train_generator.n:
        break

# Load and preprocess testing data
for images, labels in test_generator:
    X_test.append(images.reshape(images.shape[0], -1))
    y_test.append(labels.argmax(axis=1))
    if len(X_test) * 32 >= test_generator.n:
        break

# Convert lists to numpy arrays
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

# Train the model
cnne_train = lgb.Dataset(X_train, y_train)
cnne_test = lgb.Dataset(X_test, y_test)


# # Evaluate the Model

# In[36]:


import sys
from io import StringIO
import numpy as np

original_stdout = sys.stdout
sys.stdout = StringIO()

# Define model parameters with suppressed output
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_error',
    'verbose_eval': False  
}

# Train the model
cnne_model = lgb.train(params, cnne_train, num_boost_round=100, valid_sets=[cnne_test])

# Restore standard output
sys.stdout = original_stdout


# In[37]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tabulate import tabulate

# Calculate and print evaluation metrics
CNN_predicted_classes = np.argmax(cnne_model.predict(X_test, num_iteration=cnne_model.best_iteration), axis=1)

# Print CNN model evaluation metrics
print("\nCNN Model Evaluation Metrics:")
print("Loss:", CNN_loss)
print("Accuracy:", CNN_accuracy)
# Print classification report
CNN_classification_report = classification_report(y_test, CNN_predicted_classes)
print("\nCNN Model Classification Report:")
print(CNN_classification_report)


# ## Visualization of Predictions (Confusion Matrix)

# In[38]:


# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# In[39]:


# Visualization of Predictions (Confusion Matrix)
def plot_confusion_matrix(true_labels, predicted_labels, title='Confusion Matrix'):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# ## Plot confusion matrix for CNN Model

# In[40]:


plot_confusion_matrix(y_test, CNN_predicted_classes, title='CNN Model Confusion Matrix')


# - The diagonal elements of the confusion matrix represent the correct classifications, where true labels match predicted labels.
# - Off-diagonal elements represent misclassifications. The higher the values off the diagonal, the more misclassifications occurred.
# - The color intensity in the heatmap indicates the number of instances for each cell, with darker colors representing higher counts.
# - Ideally, the main diagonal should be bright, indicating correct classifications, while other areas should be relatively dim, indicating fewer misclassifications.

# In[41]:


import plotly.express as px

# Create confusion matrices for the CNN models
cnne_conf_matrix = confusion_matrix(y_test, CNN_predicted_classes)

# Plot confusion matrices using Plotly
fig_lgb = px.imshow(cnne_conf_matrix, labels=dict(x="Predicted", y="True", color="Count"), x=['Mild', 'Moderate', 'No', 'Severe'], y=['Mild', 'Moderate', 'No', 'Severe'], title='CNN Model Confusion Matrix')
fig_lgb.show()









