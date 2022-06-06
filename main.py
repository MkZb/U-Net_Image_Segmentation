# Попередні налаштування

import datetime
import os
import sys
import cv2
import numpy as np
from random import randint
from enum import Enum
from PIL import Image
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout, Rescaling
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QApplication, QLabel, QGridLayout, QFileDialog
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt
from patchify import patchify
from tqdm import tqdm

# Директорії

dt = str(datetime.datetime.now()).replace(".", "_").replace(":", "_")
model_img_save_path = f"{os.getcwd()}/models/segmentation_{dt}.png"
model_save_path = f"{os.getcwd()}/models/segmentation_model_{dt}.hdf5"
cp_path = os.getcwd() + "/models/segmentation_model-checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
csv_log_path = rf"{os.getcwd()}/logs/log_{dt}.csv"
with open(csv_log_path, 'w') as f:
    pass

# Шлях до датасету
dir = r".\dataset"


# Перелік RGB кодів кольорів маски
class MasksColors(Enum):
    Building = (60, 16, 152)
    Land = (132, 41, 246)
    Road = (110, 193, 228)
    Vegetation = (254, 221, 58)
    Water = (226, 169, 41)
    Unlabelled = (155, 155, 155)


# Попередньа обробка зображень

def load_crop_patch(dir, patch_size):
    """
    :param dir: шлях до зображень
    :param patch_size: розмір клаптику
    :return: список зображень
    """

    # Список зображень
    images = []

    for file_n, path in tqdm(enumerate(sorted(os.listdir(dir)))):
        if path.split(".")[-1] == "jpg" or path.split(".")[-1] == "png":

            # Шлях до поточного зображення
            img_path = rf"{dir}/{path}"

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cropped_x = (img.shape[1] // patch_size) * patch_size
            cropped_y = (img.shape[0] // patch_size) * patch_size

            # Поточне зображення
            img = Image.fromarray(img)

            img = np.array(img.crop((0, 0, cropped_x, cropped_y)))

            patched_img = patchify(img, (patch_size, patch_size, 3), step=patch_size)

            for j in range(patched_img.shape[0]):
                for k in range(patched_img.shape[1]):
                    patch = patched_img[j, k]
                    images.append(np.squeeze(patch))
    return images


# Підготовка зображень

def prepare_data(root, patch_size=160):
    """
    :param root: кореневий шлях датасету
    :param patch_size: розмірність клаптику
    :return: кортеж масивів зображень та їх масок
    """

    # Ініціалізація спсиків зображень та їх масок
    images = []
    masks = []

    for path, dir, files in os.walk(root):
        for subdir in dir:
            if subdir == "masks":
                masks.extend(load_crop_patch(os.path.join(path, subdir), patch_size=patch_size))
            elif subdir == "images":
                images.extend(load_crop_patch(os.path.join(path, subdir), patch_size=patch_size))

    return np.array(images), np.array(masks)


def mask_encode_to_int(masks, classes_amount=6):
    """
    :param masks: масив з масок, розбитих на клаптики
    :param num_classes: кількість класів
    :return: унітарно закодовані маски
    """

    # Список масок, закодованих цілими числами
    int_masks = []

    for mask in tqdm(masks):
        img_h, img_w, img_c = mask.shape
        int_img = np.zeros((img_h, img_w, 1)).astype(int)

        for i, label in enumerate(MasksColors):
            int_img[np.all(mask == label.value, axis=-1)] = i

        int_masks.append(int_img)

    one_hot_encoded_masks = to_categorical(y=int_masks, num_classes=classes_amount)
    return one_hot_encoded_masks


# Архітектура мережі

def unet_arch(shape):
    """
    :param shape: розміри вхідного зображення
    :return: модель з попередньо розробленою архітектурою U-Net
    """

    input_layer = Input(shape=shape)

    # Нормування значень RGB каналів
    rescaled = Rescaling(scale=1. / 255, input_shape=(img_h, img_w, img_c))(input_layer)
    prev = rescaled

    encoder = {}
    for i in [16, 32, 64, 128]:
        layer = Conv2D(i, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(prev)
        layer = Dropout(0.1)(layer)
        layer = Conv2D(i, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
        encoder[f'conv{i}'] = layer
        layer = MaxPooling2D((2, 2))(layer)
        prev = layer

    layer5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(prev)
    layer5 = Dropout(0.2)(layer5)
    layer5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(layer5)
    prev = layer5

    for i in ([128, 64, 32, 16]):
        layer = Conv2DTranspose(i, (2, 2), strides=(2, 2), padding='same')(prev)
        layer = concatenate([layer, encoder[f'conv{i}']])
        layer = Conv2D(i, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
        layer = Dropout(0.2)(layer)
        layer = Conv2D(i, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(layer)
        prev = layer

    output_layer = Conv2D(filters=6, kernel_size=(1, 1), activation="softmax")(prev)

    return Model(inputs=input_layer, outputs=output_layer)


# Оцінювання навчання

# Коефіцієнт Джаккара
def jaccard_index(ground_truth, prediction):
    """
    :param ground_truth: очікувана сегментація
    :param prediction: передбачена сегментація
    :return: коефіцієнт Джаккара
    """
    ground_truth_flat = K.flatten(ground_truth)
    prediction_flat = K.flatten(prediction)
    intersection = K.sum(ground_truth_flat * prediction_flat)
    jaccard = (intersection + 1.0) / (K.sum(ground_truth_flat) + K.sum(prediction_flat) - intersection + 1.0)
    return jaccard


# Завантаження даних

Img, Masks = prepare_data(dir, patch_size=160)

n, img_h, img_w, img_c = Img.shape
print('Загальна кількість клаптиків:', n)
show_amount = 3
random_index = [randint(0, n) for i in range(show_amount)]
sample_images = [x for z in zip(list(Img[random_index]), list(Masks[random_index])) for x in z]
Masks = mask_encode_to_int(Masks)

# Розділення датасету для навчання та валідації
Img_train, Img_test, Masks_train, Masks_test = train_test_split(Img, Masks, test_size=0.10, random_state=2)


# Передбачення

def rgb_mask(mask):
    """
    :param mask: маска сегментації
    :return: RGB маска
    """

    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i, label in enumerate(MasksColors):
        rgb_image[(mask == i)] = np.array(label.value)
    return rgb_image


# Вікна програми

class DisplayResult(QWidget):
    def __init__(self, test_img, rgb_ground_truth, rgb_image):
        self.imgs = [test_img, rgb_ground_truth, rgb_image]
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        self.setWindowTitle('Результат передбачення')
        label_original = QLabel(self)
        label_ground_truth = QLabel(self)
        label_prediction = QLabel(self)
        label_original.setText("Оригінал")
        label_ground_truth.setText("Очікуваний результат")
        label_prediction.setText("Передбачення")

        grid.addWidget(label_original, 0, 0, alignment=Qt.AlignCenter)
        grid.addWidget(label_ground_truth, 0, 1, alignment=Qt.AlignCenter)
        grid.addWidget(label_prediction, 0, 2, alignment=Qt.AlignCenter)

        for i in range(len(self.imgs)):
            height, width, channel = self.imgs[i].shape
            bytesPerLine = channel * width
            qImg = QImage(self.imgs[i].data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap01 = QPixmap.fromImage(qImg)
            pixmap_image = QPixmap(pixmap01)
            label_imageDisplay = QLabel(self)
            label_imageDisplay.setPixmap(pixmap_image)
            label_imageDisplay.setScaledContents(True)
            label_imageDisplay.setMinimumSize(1, 1)
            grid.addWidget(label_imageDisplay, 1, i)
            label_imageDisplay.show()


class DisplayExample(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        self.setWindowTitle('Приклад випадкової вибірки з датасету')
        for i in range(len(sample_images)):
            height, width, channel = sample_images[i].shape
            bytesPerLine = channel * width
            qImg = QImage(sample_images[i].data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap01 = QPixmap.fromImage(qImg)
            pixmap_image = QPixmap(pixmap01)
            label_imageDisplay = QLabel(self)
            label_imageDisplay.setPixmap(pixmap_image)
            label_imageDisplay.setScaledContents(True)
            label_imageDisplay.setMinimumSize(1, 1)
            grid.addWidget(label_imageDisplay, 0 if i % 2 == 0 else 1, int(i / 2))
            label_imageDisplay.show()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.res = []

    def initUI(self):
        btn1 = QPushButton('Навчити нову модель', self)
        btn1.setFixedSize(250, 100)
        btn1.setFont(QFont('Times', 16))
        btn1.move(25, 50)
        btn1.clicked.connect(self.train_new_model_btn)

        btn2 = QPushButton('Завантажити модель', self)
        btn2.setFixedSize(250, 100)
        btn2.setFont(QFont('Times', 16))
        btn2.move(25, 200)
        btn2.clicked.connect(self.load_model_btn)
        self.setFixedSize(300, 350)
        self.setWindowTitle('Сегментація зображень')

        self.w = DisplayExample()
        self.w.show()

    def train_new_model_btn(self):
        global model
        # Побудова моделі
        model = unet_arch(shape=(img_h, img_w, img_c))
        model.summary()

        # Збереження моделі з найбілшою точністю
        checkpoint = ModelCheckpoint(cp_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")

        # Зупинка при деградації точності протягом 2 поколінь
        stop = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="min")

        # Логгер
        csv_logger = CSVLogger(csv_log_path, separator=",", append=False)

        # Створення моделі
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", jaccard_index])

        # Тренування та збереження моделі
        model.fit(Img_train, Masks_train, epochs=20, batch_size=32, validation_data=(Img_test, Masks_test),
                  callbacks=[checkpoint, csv_logger], verbose=1)
        model.save(model_save_path)
        print("Модель збережено:", model_save_path)
        self.predict()

    def load_model_btn(self):
        model_path = QFileDialog.getOpenFileName(self, 'Оберіть модель', '.')[0]
        global model
        model = load_model(
            model_path,
            custom_objects={'jaccard_index': jaccard_index}
        )
        self.predict()

    def predict(self):
        for i in range(10):
            img_num = randint(0, len(Img_test))
            valid_img = Img_test[img_num]
            ground_truth = np.argmax(Masks_test[img_num], axis=-1)
            prediction = np.squeeze(model.predict(np.expand_dims(valid_img, 0)))
            predicted_img = np.argmax(prediction, axis=-1)
            rgb_image = rgb_mask(predicted_img)
            rgb_image = rgb_image.astype(np.uint8)
            rgb_ground_truth = rgb_mask(ground_truth)
            rgb_ground_truth = rgb_ground_truth.astype(np.uint8)
            self.res.append(DisplayResult(valid_img, rgb_ground_truth, rgb_image))
            self.res[i].show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wnd = Window()
    wnd.show()
    sys.exit(app.exec_())
