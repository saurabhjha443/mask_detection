import cv2
import numpy as np
import datetime
import tensorflow.keras
from PIL import ImageOps
import PIL.Image as Image
import os
#from predict import predict_image


model = tensorflow.keras.models.load_model('keras_model.h5')
#model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class predict_image(object):
    """
    docstring
    """
    def __init__(self,object):
        pass

    def process_and_predict(object):
        """
        docstring
        """
        if os.path.exists(object):
            # Replace this with the path to your image
            try:
                image = Image.open(object)
            except :
                return (0,str("These image is not allowed, Please try different images"))                
                
            #initialize data
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            #resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            #turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = model.predict(data)[0][0]
            
            if prediction> 0.5 :
                #cv2.putText(image,'MASK Cheers to your Awareness.',(20,420),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                label="Mask Found"
            else:
                label="No Mask Found"
                #cv2.putText(image,'NO MASK Found, Please Wear a Mask',(20,420),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            
            #datet=str(datetime.datetime.now())
            #cv2.putText(image,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            
            print(prediction)
            
            #_, jpeg = cv2.imencode('.jpg', image)
            return (prediction*100,label)
        else:
            print("File does not exist")

            

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        cv2.imwrite('temp.png',frame)
        
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open('temp.png')

        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        #image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        #prediction = predict.predict_and_return(data)
        #print(prediction)
        if model.predict(data)[0][0]> 0.5 :
        #if 1>0:
            cv2.putText(frame,'MASK Cheers to your Awareness.',(20,420),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        else:
            cv2.putText(frame,'NO MASK Found, Please Wear a Mask',(20,420),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        
        datet=str(datetime.datetime.now())
        cv2.putText(frame,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        """
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -start_time).total_seconds()
        fps = num_frames / elapsed_time      
        """
        #cv2.imshow('img',img)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()