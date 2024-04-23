import numpy as np
from flask import Flask, jsonify, request
from keras import models
from pymongo import MongoClient

app = Flask(__name__)

MONGODB_CONNECTION_STRING = "mongodb+srv://raincastid:rcastdb@rcastcluster.ef90av6.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGODB_CONNECTION_STRING)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.raincastdb

@app.route('/predictModel', methods=["POST"])
def PredictWithModel():
    hasil = 80
    return jsonify({"result": "success", "data": hasil})

@app.route('/predictModelAPI', methods=["POST"])
def PredictWithModelAPI():
    cuacaTerkini = list(request.get_json())

    # Pengambilan data minmax dari mongoDB
    data_minmax = list(db.dataminmax.find({},{'_id':False}))

    listInput = []
    for cuaca in cuacaTerkini:
        dataNormal = [
            (float(cuaca['suhu']) - data_minmax[0]['x0min'])/(data_minmax[0]['x0max'] - data_minmax[0]['x0min']),
            (float(cuaca['kelembaban']) - data_minmax[0]['x1min'])/(data_minmax[0]['x1max'] - data_minmax[0]['x1min']),
            (float(cuaca['kecepatan']) - data_minmax[0]['x2min'])/(data_minmax[0]['x2max'] - data_minmax[0]['x2min']),
            (float(cuaca['tekanan']) - data_minmax[0]['x3min'])/(data_minmax[0]['x3max'] - data_minmax[0]['x3min'])]
        listInput.append(dataNormal)

    input = np.array(listInput)

    model = models.load_model('static/predict-model/PCH-model5.keras')

    output = model.predict(input)

    curahhujan = (data_minmax[0]['ymax'] - data_minmax[0]['ymin']) * output + data_minmax[0]['ymin']

    dataWaktu = []
    for cuaca in cuacaTerkini:
        waktu = cuaca['waktu']
        dataWaktu.append(waktu)
        
    resultData = []
    for waktu, output in zip(dataWaktu, curahhujan):
        hasil = float(output[0])
        if output[0] < 0:
            hasil = 0
        resultData.append({
            'waktu': waktu,
            'hasil': hasil
        })
    return jsonify({"result": "success", "data": resultData})

if __name__ == '__main__':  
   app.run('0.0.0.0',port=5000,debug=True)