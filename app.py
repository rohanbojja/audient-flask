from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import werkzeug
import scipy.optimize
import os,pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa

app = Flask(__name__) 
api = Api(app) 

class receiveWav(Resource): 
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        audioFile = args['file']
        audioFile.save("song.mp3")
        scaler = pickle.load(open("scaler.ok","rb"))
        x , sr = librosa.load("song.mp3",mono=True,duration=5)
        y=x
        #Extract the features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rmse = librosa.feature.rmse(y=y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            features += f' {np.mean(e)}'
        input_data2 = np.array([float(i) for i in features.split(" ")]).reshape(1,-1)
        input_data2 = scaler.transform(input_data2)
        return jsonify(input_data2.tolist())
  
  
api.add_resource(receiveWav, '/receiveWav/sav') 
  
  
# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True) 