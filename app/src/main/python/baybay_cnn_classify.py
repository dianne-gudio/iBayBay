import tensorflow as tf
import numpy as np
import cv2
import base64
from os.path import dirname, join

# Baybayin Classes
baybay_list = ['a', 'e_i', 'o_u',
        'b', 'ba', 'be_bi', 'bo_bu', 
        'd', 'da_ra', 'de_di', 'do_du', 
        'r', 'ra', 're_ri', 'ro_ru',
        'g', 'ga', 'ge_gi', 'go_gu', 
        'h', 'ha', 'he_hi', 'ho_hu',
        'k', 'ka', 'ke_ki', 'ko_ku',
        'l', 'la', 'le_li', 'lo_lu',
        'm', 'ma', 'me_mi', 'mo_mu',
        'n', 'na', 'ne_ni', 'no_nu',
        'ng', 'nga', 'nge_ngi', 'ngo_ngu',
        'p', 'pa', 'pe_pi', 'po_pu',
        's', 'sa', 'se_si', 'so_su',
        't', 'ta', 'te_ti', 'to_tu',
        'w', 'wa', 'we_wi', 'wo_wu',
        'y', 'ya', 'ye_yi', 'yo_yu']

### Getting the Dataset (INPUT)
def get_data(encoded_img): # Image Pre-Processing
    # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) 
    decoded_data = base64.b64decode(encoded_img)
    np_data = np.fromstring(decoded_data,np.uint8)

    img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] # convert image to black and white pixels
    img = cv2.resize(img, (28,28))

    return np.array(img).reshape(-1,28,28,1)

### Loading and Using the CNN Model (PROCESS)
def load_model():
    model_path = join(dirname(__file__), "baybay_cnn.h5") # path to saved model

    return tf.keras.models.load_model(model_path) # loading CNN Model

def load_model_weights():
    model_weights_path = join(dirname(__file__), 'baybay_cnn_weights.h5')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28,28,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Dense(63, activation='softmax'),
    ])

    model.load_weights(model_weights_path) # Load CNN Model

    return model

# print('Prediction: {} [{}]'.format(pred, pred2))

### Visualization of Predictions (OUTPUT)
def get_prediction(encoded_img): # returns classification after prediction
    image_data = get_data(encoded_img)
    # model = load_model()
    model = load_model_weights()

    y_pred = model.predict(image_data)

    pred = np.argmax(y_pred)
    pred2 = np.argsort(-y_pred, axis=1)[:,1][0]

    return baybay_list[pred]