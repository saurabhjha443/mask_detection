import cv2 
import numpy as np
import datetime
import tensorflow.keras
from PIL import Image, ImageOps


def predict_and_return( frame):
    """
    docstring
    """
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
    prediction = model.predict(data)
    print(prediction)


    
    return prediction









if __name__ == "__main__":

    model = tensorflow.keras.models.load_model('keras_model.h5')


    cap=cv2.VideoCapture(0)
    start_time = datetime.datetime.now()
    num_frames=0

    while cap.isOpened():
        _,img=cap.read()    
        if predict_and_return(img)[0][0] > 0.65 :
            cv2.putText(img,'MASK Cheers to your Awareness.',(20,420),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        else:
            cv2.putText(img,'NO MASK Found, Please Wear a Mask',(20,420),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -start_time).total_seconds()
        fps = num_frames / elapsed_time      
        
        cv2.imshow('img',img)
        
        if cv2.waitKey(1)==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()