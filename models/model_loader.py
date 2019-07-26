import keras
import joblib

__cach = dict()

def pip_provider():
    if 'pipline' in __cach:
        return __cach['pipline']
    else:
        __cach['pipline'] = joblib.load('models/pipline') 

    return __cach['pipline']

def small_NN_proider():
    if 'Small_conc' in __cach:
        return __cach['Small_conc']
    else:
        __cach['Small_conc'] = keras.models.load_model('models/Small_conc.h5')
                   
    return __cach['Small_conc']

def large_NN_proider():
    if 'Large_conc' in __cach:
        return __cach['Large_conc']
    else:
        __cach['Large_conc'] = keras.models.load_model('models/Large_conc.h5')
                   
    return __cach['Large_conc']

