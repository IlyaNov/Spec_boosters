import numpy as np
from models.model_loader import pip_provider, small_NN_proider, large_NN_proider
import processing.image_processers as ip

def predict(path):
    pipline , small_NN, large_NN = pip_provider(), small_NN_proider(), large_NN_proider()   
    try:  
        saver_large = ip.make_saver(ip.spec_builder,save_to_file=False)
        saver_small = ip.make_saver(ip.spec_builder_small,save_to_file=False)
        transformer_small = ip.make_transformer(saver_small,path=None)
        transformer_large = ip.make_transformer(saver_large,path=None)

        small_im = np.array(transformer_small(path))
        large_im = np.array(transformer_large(path))

        small_pred = small_NN.predict(small_im)
        large_pred = large_NN.predict(large_im)

        concatenated = np.concatenate((large_pred,small_pred),axis=1)

        predictions = pipline.predict_proba(concatenated)[:,1]
        return np.mean(predictions)

    except Exception as e:
            print("Error with getting feature from %s: %s" % (path, str(e)))
            return 0

    
