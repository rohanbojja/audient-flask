from flask import Flask, jsonify, request
import scipy.optimize
import os,pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
import logging
import soundfile as sf
from pydub import AudioSegment
import subprocess as sp
import ffmpeg
from io import BytesIO

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__) 

@app.route('/receiveWav',methods = ['POST'])
def upload():
    if(request.method == 'POST'):
        f = request.files['file']
        app.logger.info(f'AUDIO FORMAT\n\n\n\n\n\n\n\n\n\n: {f}')
        proc = (
            ffmpeg.input('pipe:',format='mp4')
            .output('pipe:', format='aiff')
            .run_async(pipe_stdin=True,pipe_stdout=True, pipe_stderr=True)
        )
        audioFile,err = proc.communicate(input=f.read())
        audioFile =  BytesIO(audioFile)
        scaler = pickle.load(open("scaler.ok","rb"))
        x , sr = librosa.load(audioFile,mono=True,duration=5)
        y=x
        #Extract the features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rmse = librosa.feature.rms(y=y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            features += f' {np.mean(e)}'
        input_data2 = np.array([float(i) for i in features.split(" ")]).reshape(1,-1)
        input_data2 = scaler.transform(input_data2)
        return jsonify(input_data2.tolist())
  
# driver function 
if __name__ == '__main__':   
    app.run(debug = True) 