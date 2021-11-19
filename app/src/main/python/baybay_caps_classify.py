import numpy as np
import cv2
import base64
from os.path import dirname, join

import baybay_capsnet as bcaps

model_name = 'Baybayin'

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

### Getting the Data / Pre-Processing (INPUT)
def get_data(encoded_img): # Image Pre-Processing
    # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) 
    decoded_data = base64.b64decode(encoded_img)
    np_data = np.fromstring(decoded_data,np.uint8)

    img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] # convert image to black and white pixels
    img = cv2.resize(img, (28,28))

    img = np.array(img).reshape(-1,28,28)
    return (img / 256)[...,None].astype('float32')

def model_summary():
    model = load_model()
    print(model.model.summary())

### Loading and Using the CapsNet Model (PROCESS)
def load_model():
    model_path = join(dirname(__file__), 'baybay_caps.h5') # path to saved model weights

    model = bcaps.EfficientCapsNet(model_name, verbose=False, custom_path=model_path) # loading CapsNet Model
    model.load_graph_weights()

    return model

# print('Prediction: {} [{}]'.format(pred, pred2))

### Visualization of Predictions (OUTPUT)
def get_prediction(encoded_img):
    img_data = get_data(encoded_img)
    
    model = load_model()

    y_pred, _ = model.predict(img_data)

    pred = np.argmax(y_pred)
    pred2 = np.argsort(-y_pred, axis=1)[:,1][0]

    return baybay_list[pred]