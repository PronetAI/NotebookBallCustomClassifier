import os
import cv2
import keras
from google.colab.patches import cv2_imshow
import numpy as np
from sklearn.model_selection import train_test_split
data, labels = [],[]
main_folder="images"
sub_folders=os.listdir(main_folder)
# print(sub_folders)
sub_folders.remove(".ipynb_checkpoints")
# print(sub_folders)
for folder in sub_folders:
  # print(folder)
  path_to_folder=os.path.join(main_folder,folder)
  # print(path_to_folder)
  folder_dir=os.listdir(path_to_folder)
  if ".ipynb_checkpoints" in folder_dir:
    folder_dir.remove(".ipynb_checkpoints")
  for img in folder_dir:
    full_path_to_image=os.path.join(path_to_folder, img)
    # print(full_path_to_image)
    image=cv2.imread(full_path_to_image)
    resized_image=cv2.resize(image,(50,50))
    gray_image=cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    data.append(gray_image)
    if folder=="ball":
      labels.append(0)
    elif folder == "notebook":
      labels.append(1)
    # cv2_imshow(gray_image)
# print(len(data), len(labels))
data_array=np.array(data)
labels_array=np.array(labels)
# print(data_array.shape,labels_array.shape)
train_images, test_images, train_labels, test_labels = train_test_split(data_array, labels_array, test_size = 0.2)
# print(train_images.shape,test_images.shape)

train_images=train_images/255
test_images=test_images/255

# Building the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50,50)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax'),
])

# Compiling the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fitting the model
model.fit(train_images, train_labels, epochs=30)

