import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import io 
from PIL import Image 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

image_folder = '../DatasetwithR/'
#image_folder = '../DatasetRGB/'
#image_folder = '../Dataset333/BOTH/'

Directorylist = []
PolarTImages = []
HistofPolor = []
HistofPolorImg = []

radius = 110  
angle = 35

def loaddataset():
    print("load dataset :")
    
    Directorylist = []
    for pathb in os.listdir(image_folder):
        Directorylist.append(pathb)
    print("Classes found:", Directorylist)

    imageslist = []
    labels = []
    

    for class_idx, class_name in enumerate(Directorylist):
        class_path = os.path.join(image_folder, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith(".png"):
                imageslist.append(os.path.join(class_path, filename))
                labels.append(class_idx)  

 
    images = [img_to_array(load_img(img_path, target_size=(240, 240))) for img_path in imageslist]
    images = np.array(images)
    images = images / 255.0
    
    # Extract features
    histpolar = Extractfeaures(images)
    
   
    labels = to_categorical(labels, num_classes=3)
    
    print("load completed :")
    print(f"Features shape: {histpolar.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return histpolar, labels

def Extractfeaures(images):
    print("Extractfeaures :")
    features_list = []

    for image in images:
        image = img_as_float(image)
        image_polar = warp_polar(image, radius=radius, channel_axis=-1)
        PolarTImages.append(image_polar)

        n, bins, patches = plt.hist(image_polar.ravel(), 256, [0, 1])
        plt.close()  # Closing to prevent memory leak
        
        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        r = n  
        theta = bin_midpoints * 2 * np.pi 
        
        feature_vector = np.zeros((6, 256))

        

        angle1_min = np.deg2rad(35)
        angle1_max = np.deg2rad(135)
        angle1_mask = (theta >= angle1_min) & (theta <= angle1_max)
        feature_vector[0, angle1_mask] = r[angle1_mask]

        # Filter 2: 135° to 225° (diagonal/left orientation)
        angle2_min = np.deg2rad(200)
        angle2_max = np.deg2rad(280)
        angle2_mask = (theta >= angle2_min) & (theta <= angle2_max)
        feature_vector[1, angle2_mask] = r[angle2_mask]

        # Filter 3: 315° to 45° (wrapping around 0°, horizontal orientation)
        # Handle the wrap-around case
        angle3_min = np.deg2rad(300)
        angle3_max = np.deg2rad(360)
        angle3_mask = (theta >= angle3_min) & (theta <= angle3_max)
        feature_vector[2, angle3_mask] = r[angle3_mask]

      #  angle_min = np.deg2rad(45)   
       # angle_max = np.deg2rad(135)
       # angle_mask = (theta >= angle_min) & (theta <= angle_max)
       # feature_vector[0, angle_mask] = r[angle_mask]
        
        radius1_min = 20  
        radius1_max = 100  
        radius1_mask = (r >= radius1_min) & (r <= radius1_max)
        feature_vector[3, radius1_mask] = r[radius1_mask]
        
        radius2_min = 150  
        radius2_max = 220   
        radius2_mask = (r >= radius2_min) & (r <= radius2_max)
        feature_vector[4, radius2_mask] = r[radius2_mask]

        radius3_min = 250  
        radius3_max = 500   
        radius3_mask = (r >= radius3_min) & (r <= radius3_max)
        feature_vector[5, radius3_mask] = r[radius3_mask]



        
        features_list.append(feature_vector)
        
    return np.array(features_list)

def create_cnn_model(input_shape, num_classes=3):
    
    inputs = Input(shape=input_shape)
     
    #x = layers.Permute((2, 1))(inputs)  # Now shape: (256, 6)
     
    #x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
   # x = layers.BatchNormalization()(x)
   # x = layers.MaxPooling1D(pool_size=2)(x)  # Shape: (128, 32)
     
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # Shape: (64, 64)
     
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # Shape: (32, 128)
     
    x = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)  # Shape: (256,)
     
   # x = layers.Dense(128, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)
    #x = layers.Dense(64, activation='relu')(x)
    #x = layers.Dropout(0.3)(x)
     
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def create_simpler_cnn_model(input_shape, num_classes=3):
     
    inputs = Input(shape=input_shape)
     
    x = layers.Reshape((input_shape[0] * input_shape[1], 1))(inputs)  # Shape: (512, 1)
     
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
     
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)


def create_mlp_model(input_shape, num_classes=3):
     
    inputs = Input(shape=input_shape)
     
    x = layers.Flatten()(inputs)  # Shape: (512,)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)
 


input_shape = (6, 256)  # histo polar
X, y = loaddataset()
print(f"X shape: {X.shape}, y shape: {y.shape}")
 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {x_train.shape}, {y_train.shape}")
print(f"Test set: {x_test.shape}, {y_test.shape}")
 
model = create_cnn_model(input_shape, num_classes=3)
#model = create_simpler_cnn_model(input_shape, num_classes=3)

#model.compile(
   # optimizer='adam',
  #  loss='categorical_crossentropy',
 #   metrics=['accuracy']
#)
import time
start_time = time.time()
print("Train started  ...")

status=1

if  status==1: 
        model.compile(
        optimizer='SGD',
        loss='categorical_crossentropy',
      #  loss='mean_absolute_error',
        metrics=['accuracy'])

elif  status==2: 
        model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

elif  status==3: 
        model.compile(
        optimizer='RMSprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
elif  status==4: 
        model.compile(
        optimizer='Adagrad',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
elif  status==5: 
        model.compile(
        optimizer='Adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

print("\nModel Summary:")
model.summary()
 
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)

# Train model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


end_time = time.time()
 
duration = end_time - start_time
print(f"training  duration ={duration:.4f} seconds")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot learning rate
if 'lr' in history.history:
    plt.subplot(1, 3, 3)
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Print final accuracy
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Final training accuracy: {train_acc:.4f}")
print(f"Final validation accuracy: {val_acc:.4f}")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save model
model.save("polar_classification_model.keras")

# Make predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
#print(classification_report(true_classes, predicted_classes, target_names=Directorylist))

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=Directorylist, yticklabels=Directorylist)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\nSample Predictions (first 10 test samples):")
for i in range(min(10, len(x_test))):
    true_label = Directorylist[true_classes[i]]
    pred_label = Directorylist[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]]
    print(f"Sample {i+1}: True: {true_label}, Predicted: {predicted_classes[i]} ({pred_label}), Confidence: {confidence:.3f}")