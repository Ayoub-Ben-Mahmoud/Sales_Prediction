from flask import Flask, request, jsonify, render_template
from flask_cors import CORS,cross_origin
import pickle
import numpy as np


app=Flask(__name__)
model = pickle.load(open('xgbmodel.pkl','rb'))
scaler = pickle.load(open('scaler.pickle','rb'))

app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
@app.errorhandler(404)

def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''For rendering results on HTML GUI'''
    features = [float(x) for x in request.form.values()]
    pre_final_features = [np.array(features)]
    final_features = scaler.transform(pre_final_features)
    prediction = model.predict(final_features)
    print('Prediction Value is',prediction[0])
    output=prediction[0]

    return render_template('index.html',prediction_text='The predicted Sales value is about :  {} $'.format(output))

    if __name__ == "__main__":
        app.run(debug=True)