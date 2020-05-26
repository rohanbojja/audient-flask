from flask import Flask, jsonify, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
import logging
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np


# Load necessary elements
loc = "saved_models_best"
model = tf.keras.models.load_model(loc,
custom_objects={'KerasLayer':hub.KerasLayer})
scaler = pickle.load(open("scaler_best.ok","rb"))


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/getPredictions',methods = ['POST'])
def upload2():
    if(request.method == 'POST'):
    
        f = request.files['file']
        dur = int(request.form["dur"]) # Duration, to be sent 
        label_code = int(request.form["label_code"])
        
        audioFile =  f
        
        ret_list = []
        
        t=0
        while(t<dur-1):
                audioFile.seek(0)
                x , sr = librosa.load(audioFile,offset=t,mono=True,duration=5)
                y=x
                #Extract the features
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                rmse = librosa.feature.rms(y=y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                tempogram = librosa.feature.tempogram(y=y, sr=sr)
                bpm = librosa.beat.tempo(y=y, sr=sr)
                feat_arrays =[chroma_stft, spec_cent, spec_bw, rolloff, zcr, rmse, tempogram,bpm ]
                k = ["_mean", "_median", "_sd", "_ptp", "_kurt", "_skew"]
                to_append = ""
                for i in feat_arrays:
                    to_append += f' {np.mean(i)} {np.median(i)} {np.std(i)} {np.ptp(i)}'  
                
                for i in mfcc:
                    to_append += f' {np.mean(i)} {np.median(i)} {np.std(i)} {np.ptp(i)}'
                    
                #app.logger.info(f'{len(to_append.split(" "))}')
                to_append2 = to_append.split(" ")[1:]
                input_data2 = np.array([float(i) for i in to_append2]).reshape(1,-1)
                
                #Handle label_code here
                
                input_data2 = scaler.transform(input_data2)
                tf_model_predictions = model.predict(input_data2)
                
                genres = "Blues Classical Country Disco Hiphop Jazz Metal Pop Reggae Rock".split()
                
                
                high=0
                tf_model_predictions = tf_model_predictions[0]
                res = {}
                for i,e in enumerate(tf_model_predictions):
                        res[genres[i]] = str(e)
                ret_list.append(res)
                t+=5
        return jsonify(ret_list)
  


@app.route('/getFeatures',methods = ['POST'])
def upload():
    if(request.method == 'POST'):
        f = request.files['file']
        dur = int(request.form["dur"]) # Duration, to be sent 
        audioFile =  f
        ret_list = []
        x , sr = librosa.load(audioFile,mono=True,duration=dur)
        y=x
        #Extract the features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rmse = librosa.feature.rms(y=y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        bpm = librosa.beat.tempo(y=y, sr=sr)
        feat_arrays =[chroma_stft, spec_cent, spec_bw, rolloff, zcr, rmse, tempogram,bpm ]
        #for stat in stats:
        k = ["_mean", "_median", "_sd", "_ptp", "_kurt", "_skew"]
        to_append = ""
        for i in feat_arrays:
            to_append += f' {np.mean(i)} {np.median(i)} {np.std(i)} {np.ptp(i)}'  
        
        for i in mfcc:
            to_append += f' {np.mean(i)} {np.median(i)} {np.std(i)} {np.ptp(i)}'
            
        #app.logger.info(f'{len(to_append.split(" "))}')
        to_append2 = to_append.split(" ")[1:]
        input_data2 = np.array([float(i) for i in to_append2]).reshape(1,-1)
        input_data2 = scaler.transform(input_data2)
        
        return jsonify(input_data2.tolist())
  
# driver function 
if __name__ == '__main__':   
    app.run(debug = True, host='0.0.0.0') 
