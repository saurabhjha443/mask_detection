"""import tensorflow.keras
from PIL import Image, ImageOps
"""
"""
For Predict Class Reference

"""

"""



class predict_image(object):

    def __init__(self):
        # load model from JSON file
        self.model = tensorflow.keras.models.load_model('keras_model.h5')
    
    
    def predict_and_return(self,frame):
        
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
        self.prediction = self.model.predict(data)
        print(self.prediction)
        
        return self.prediction"""