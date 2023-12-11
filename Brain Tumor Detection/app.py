from flask import Flask,request,render_template
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.utils import load_img,img_to_array
import os
from werkzeug.utils import secure_filename


app=Flask(__name__)



def model_predict(image_path):
    model=load_model('brain_resnet.h5')
    image = load_img(image_path,target_size=(256,256))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))  
    print("Predicted")
    
    if result == 0:
        return "No Tumor"       
    elif result == 1:
        return "Tumor"
    
    

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename

    imagefile.save(image_path)

    print(image_path)
    
    final=model_predict(image_path)

    


    return render_template('index.html',result=final)



if (__name__) == "__main__":
    app.run(port=5000,debug=True)  