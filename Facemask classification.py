
import cv2, os

data_path = 'dataset'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories, labels))

print(label_dict)
print(categories)
print(labels)
img_size = 100
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            target.append(label_dict[category])

        except Exception as e:
            print('Exception:', e)
import numpy as np

data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

from keras.utils import np_utils

new_target = np_utils.to_categorical(target)
np.save('data', data)
np.save('target', new_target)
import numpy as np

data = np.load('data.npy')
target = np.load('target.npy')
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
history = model.fit(train_data, train_target, epochs=15, callbacks=[checkpoint], validation_split=0.2)
import tensorflow_addons as tfa

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.Accuracy(),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tfa.metrics.F1Score(num_classes=nb_classes, average='macro', threshold=0.5)])

from google.colab import drive

drive.mount('/content/drive')

import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Activation, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

images = []
labels = []

i = 0
Size = 224
folder_range = ['00000']

for each in folder_range:
    currentFolder = '/content/drive/Shared drives/270 DATA/CMFD/' + each
    for i, file in enumerate(os.listdir(currentFolder)):
        im = cv2.imread((os.path.join(currentFolder, file)))

        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (Size, Size))
        img = resized_image[..., ::-1].astype(np.float32)
        img = preprocess_input(img)
        images.append(img)
        labels.append('Correct Masked Face')

for each in folder_range:
    currentFolder = '/content/drive/Shared drives/270 DATA/IMFD/' + each
    for i, file in enumerate(os.listdir(currentFolder)):
        im = cv2.imread((os.path.join(currentFolder, file)))

        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (Size, Size))
        img = resized_image[..., ::-1].astype(np.float32)
        img = preprocess_input(img)
        images.append(img)
        labels.append('Incorrect Masked Face')
    images = np.array(images, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, stratify=labels, random_state=42)
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
baseModel = EfficientNetV2L(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
from tensorflow.keras import layers

for layer in baseModel.layers:
    layer.trainable = False
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(1024, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(
    aug.flow(X_train, y_train, batch_size=BS),
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BS,
    epochs=EPOCHS)
Predicted = model.predict(X_test, batch_size=BS)
Predicted = model.predict(X_test, batch_size=BS)
from sklearn.metrics import classification_report

Predicted = np.argmax(Predicted, axis=1)
print(classification_report(y_test.argmax(axis=1), Predicted,
                            target_names=lb.classes_))
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from google.colab import drive

drive.mount('/content/drive')
% cd / content / drive / MyDrive / Data
270 / Data
% cd / content / drive / MyDrive / Data
270 / Data
# importing libraries
% matplotlib
inline
import tensorflow
import keras
import os
import glob
from skimage import io
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io


def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True


from skimage import io


def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return Truefrom
    skimage
    import io


def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True


from easyimages import EasyImage, EasyImageList, bbox

Li = EasyImageList.from_multilevel_folder('/content/drive/MyDrive/Data 270/Data/')
# Li.symlink_images()
# Li.html(size = 100)
Li
# Li.html()
image1 = EasyImage.from_file('/content/drive/MyDrive/Data 270/Data/00012_Mask.jpg', label=['Person'], lazy=True)
image1.show()
# Importing and Loading the data into a data frame

dataset_path = '/content/drive/MyDrive/Data 270/Data/'

# apply glob module to retrieve files/pathnames

img_path = os.path.join(dataset_path, '*')
img_path = glob.glob(img_path)
# accessing an image file from the dataset classes
image = io.imread(img_path[4])

# plotting the original image
i, (im1) = plt.subplots(1)
i.set_figwidth(15)
im1.imshow(image)
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


# Display one image
def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Display two images
def display(a, b, title1="Original", title2="Edited"):
    plt.figure(figsize=(15, 15))
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Display three images
def display_three(a, b, c, title1="Original", title2="Edited", title3="Editeds"):
    plt.figure(figsize=(15, 15))
    plt.subplot(131), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(c), plt.title(title3)
    plt.xticks([]), plt.yticks([])
    plt.show()


test_img = cv2.imread('/content/drive/MyDrive/Data 270/Data/00023_Mask.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# Remove noise
# Gaussian
no_noise = []
blur = cv2.GaussianBlur(test_img, (5, 5), 0)

display(test_img, blur, title1="Original", title2="Blurred")
plt.style.use('seaborn')
test_image = cv2.imread('/content/drive/MyDrive/Data 270/Data/00023_Mask.jpg')
dst = cv2.fastNlMeansDenoisingColored(test_image, None, 11, 6, 7, 21)
row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
axs[1].set_title('Fast Means Denoising')
plt.show()
# Segmentation
grayish = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
rett, threshh = cv2.threshold(grayish, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
edgess = cv2.dilate(cv2.Canny(threshh, 0, 255), None)
# Displaying segmented images
display_three(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), edgess, threshh, 'Blurred', 'Edges', 'Segmented')
# Segmentation
gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
# Displaying segmented images
display_three(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), edges, thresh, 'Blurred', 'Edges', 'Segmented')
# Further noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
edgesa = cv2.dilate(cv2.Canny(sure_fg, 0, 255), None)

# Displaying segmented back ground
display_three(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), edgesa, sure_bg, 'Original', 'Edges', 'Segmented Background')
import cv2


def is_valid(image):
    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.1
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.5
    return s_perc > s_thr


noise1 = cv2.cvtColor(io.imread('/content/drive/MyDrive/Data 270/Data/00023_Mask.jpg'), cv2.COLOR_RGB2BGR)
is_valid(noise1)


def unique(list1):
    x = np.array(list1)
    print(np.unique(x))


i = 0
noisy = []
currentFolder = '/content/drive/MyDrive/Data 270/Data'
for i, file in enumerate(os.listdir(currentFolder)):
    im = cv2.imread((os.path.join(currentFolder, file)))
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    noisy.append(is_valid(img))
    # print(is_valid(img))

unique(noisy)
from google.colab.patches import cv2_imshow

test_img = cv2.imread('/content/drive/MyDrive/Data 270/Data/00016_Mask.jpg')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(test_img, scaleFactor=1.2, minNeighbors=4)
n = 0
for (x, y, w, h) in eyes:
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    n += 1
cv2_imshow(test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(n)
new_img = cv2.imread('/content/drive/MyDrive/Data 270/Data/00016_Mask.jpg')
gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
n = 0
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    n += 1
    print(x + w, y + h)

for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(new_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
cv2_imshow(new
_img)
cv2.waitKey(0)
print(n)
as_img = cv2.imread('/content/drive/MyDrive/Data 270/Data/00016_Mask.jpg')
cropped_image = as_img[x:y + h, y:x + w]
cv2_imshow(cropped_image)
k = 0
incomplete_df = []
currentFolder = '/content/drive/MyDrive/Data 270/Data'
for k, file in enumerate(os.listdir(currentFolder)):
    im = cv2.imread((os.path.join(currentFolder, file)))
    resized_image = cv2.resize(im, (128, 128))
    img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img, 1.2, 4)
    eyes = eye_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4)

    n = 0
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        n += 1

    # print(n)
    if (
            n != 1):  # If no faces were detected, increment n by 1 -> Add the file name to list of possible incomeplete images
        incomplete_df.append(file)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(resized_image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2_imshow(resized_image)
    cv2.waitKey(0)

print('List of possible incomplete images: ', incomplete_df)

from google.colab import drive

drive.mount('/content/drive')
% cd / content / drive / Shared
drives / 270
DATA
import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

images = []
labels = []

i = 0
Size = 100
folder_range = ['00000', '01000', '02000', '03000', '04000', '05000', '06000', '07000', '08000', '09000', '10000',
                '11000']

for each in folder_range:
    currentFolder = '/content/drive/Shared drives/270 DATA/CMFD/' + each
    for i, file in enumerate(os.listdir(currentFolder)):
        im = cv2.imread((os.path.join(currentFolder, file)))

        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (Size, Size))
        img = resized_image[..., ::-1].astype(np.float32)
        # normalize
        img = preprocess_input(img)
        images.append(img)
        labels.append('Correct Masked Face')

for each in folder_range:
    currentFolder = '/content/drive/Shared drives/270 DATA/IMFD/' + each
    for i, file in enumerate(os.listdir(currentFolder)):
        im = cv2.imread((os.path.join(currentFolder, file)))

        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (Size, Size))
        img = resized_image[..., ::-1].astype(np.float32)
        img = preprocess_input(img)
        images.append(img)
        labels.append('Incorrect Masked Face')
    images = np.array(images, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, stratify=labels, random_state=42)
print('Training dataset contains: ', X_train.shape, ' data')
print('Testing dataset contains: ', X_test.shape, ' data')
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
X_train.shape
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(100, 100, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False
INIT_LR = 1e-4
EPOCHS = 25
BS = 32
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(
    aug.flow(X_train, y_train, batch_size=BS),
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BS,
    epochs=EPOCHS)
Predicted = model.predict(X_test, batch_size=BS)

from sklearn.metrics import classification_report

Predicted = np.argmax(Predicted, axis=1)
print(classification_report(y_test.argmax(axis=1), Predicted, target_names=lb.classes_))
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
from google.colab import drive

drive.mount('/content/drive')
% cd / content / drive / Shared
drives / 270
DATA

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

images = []
labels = []

i = 0
Size = 100
folder_range = ['01000', '02000', '03000', '04000', '05000', '06000', '07000', '08000', '09000', '10000', '11000']

for each in folder_range:
    currentFolder = '/content/drive/Shared drives/270 DATA/CMFD/' + each
    for i, file in enumerate(os.listdir(currentFolder)):
        im = cv2.imread((os.path.join(currentFolder, file)))

        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (Size, Size))
        img = resized_image[..., ::-1].astype(np.float32)
        img = preprocess_input(img)
        img = img / 255
        images.append(img)
        labels.append('Correct Masked Face')

for each in folder_range:
    currentFolder = '/content/drive/Shared drives/270 DATA/IMFD/' + each
    for i, file in enumerate(os.listdir(currentFolder)):
        im = cv2.imread((os.path.join(currentFolder, file)))

        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (Size, Size))
        img = resized_image[..., ::-1].astype(np.float32)
        img = preprocess_input(img)
        img = img / 255
        images.append(img)
        labels.append('Incorrect Masked Face')
    images = np.array(images, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
image_label1 = np.ones(len(image_arr1), dtype=np.uint8)
image_label0 = np.zeros(len(image_arr0), dtype=np.uint8)
plt.figure(figsize=(7, 7))
plt.subplot(1, 2, 1)
plt.title("Wear Mask", fontsize=10)
plt.imshow(cmfd_df[0])
plt.subplot(1, 2, 2)
plt.title("Wrong Mask Wear", fontsize=10)
plt.imshow(imfd_df[0])
# reshape（H,W）image data to single dimension H*W
X = images.reshape(len(images), -1)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)

print(f'train data: {X_train.shape}')
print(f'test data: {X_test.shape}')
svm_pca = make_pipeline(PCA(n_components=50),
                        LinearSVC(random_state=0, tol=1e-5))
svm_pca = svm_pca.fit(X_train, y_train)
y_pred = svm_pca.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='.1f', cmap='GnBu', annot_kws={"fontsize":12})
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
t = classification_report(y_test, y_pred, target_names=['0', '1'])
print(t)

acc = accuracy_score(y_test, y_pred)
print(f'accuracy score：{round(acc, 3)}')
# f1 = f1_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f'f1 score：{round(f1, 3)}')

y_pred_prob = svm_pca.decision_function(X_test)
auc = roc_auc_score(y_test, y_pred_prob)
print(f'auc score：{round(auc, 3)}')
