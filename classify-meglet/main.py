# importing library 
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

import json
from flask import Flask , request, jsonify

app = Flask(__name__)

# Load all model artifacts 

model = load_model('./vgg16_model.h5')

# data preprocessing properties 
dict1 = {0 : 'horizontal_bar', 1: 'line', 2: 'pie', 3: 'scatter', 4: 'vertical_bar' }

@app.route('/predict',methods=['GET','POST']) #decorate the prediction_endpoint func with the desired endpoint and methods
def prediction_endpoint():
    if request.method == 'GET':
        return 'kindly send a POST request'    #return this if request is 'GET'
    elif request.method == 'POST':

        input_data = request.files.get("input_file") ##

        test_image = image.load_img(input_data, color_mode ='rgb', target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        res = np.argmax(result)
        #prediction 
        response = json.dumps(dict1[res])
    return response #send the model response back to the user


if __name__ == "__main__":
    app.run()#run flask app instance 

