from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#load the model
model=load_model(r'C:\Users\jinan\project\envs\CHEST-XRY\fmodel.h5')
#preduction of the model
img=image.load_img(r'C:\Users\jinan\project\envs\CHEST-XRY\validation\PNEUMONIA\PNEUMONIA(3845).jpg', target_size=(224,224))
imagee=image.img_to_array(img) #Converting the X-Ray into pixels
imagee=np.expand_dims(imagee, axis=0)
img_data=preprocess_input(imagee)
prediction=model.predict(img_data)
if prediction[0][0]==1: #Printing the prediction of model.
  print('COVID19.')
elif prediction[0][1]==1:
  print('Person is Normal')
else:
  print('Person is affected with PNEUMONIA')
print(f'Predictions: {prediction}')
