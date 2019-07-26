import os
import glob
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import scipy
from keras import backend as K
from models.model_loader import pip_provider, small_NN_proider, large_NN_proider
import processing.image_processers as ip
from predictor import predict


config = tf.ConfigProto()
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)
tf.logging.set_verbosity(tf.logging.ERROR)

dataset_dir = "."
wav_paths = sorted(glob.glob('./Testing_Data/*.wav', recursive=True))
# wav_paths = sorted(glob.glob('../audio/Training_Data/human/*.wav', recursive=True))

result = predict(np.random.choice(wav_paths))

eval_protocol_path = "protocol_test.txt"
eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
eval_protocol.columns = ['path', 'key']
eval_protocol['score'] = 0.0

print(eval_protocol.shape)
print(eval_protocol.sample(5).head())

# for i in wav_paths:
#     score = predict(i)
#     print(score)

for protocol_id, protocol_row in tqdm.tqdm(list(eval_protocol.iterrows())):
    score = predict(os.path.join(dataset_dir, protocol_row['path']))
    eval_protocol.at[protocol_id, 'score'] = score
eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
print(eval_protocol.sample(5).head())
