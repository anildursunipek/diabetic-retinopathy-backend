import numpy as np
import cv2

class Preprocessor:
    def __init__(self, image_size: int = 224, tol:int = 7, sigmax: int = 10):
        self.image_size = image_size
        self.tol = tol
        self.sigmax = sigmax

    def preprocessing(self, image: np.array) -> np.array:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.crop_image_from_gray(image).astype('uint8')
        image = cv2.resize(image, (self.image_size, self.image_size))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , self.sigmax) ,-4 ,128)
        return image.astype('uint8') # numpy.array
    
    def crop_image_from_gray(self, img) -> np.array:
        if img.ndim ==2:
            mask = img>self.tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img>self.tol

            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0):
                return img
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img

if __name__ == "__main__":
    # TEST CODES
    preprocessor = Preprocessor(image_size=528, tol=7, sigmax=10)

    # Read an example image from sample folder
    image_path = "samples/moderate/00a8624548a9.png"
    image = cv2.imread(image_path)

    # Preprocess the image
    image = preprocessor.preprocessing(image)
    print("[INFO] Image Shape: ", image.shape)
    print("[INFO] Type Of Image: ", type(image))

    # Open the image to check result
    cv2.imshow("Sample image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



