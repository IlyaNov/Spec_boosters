import  cv2 as cv
import numpy as np


class AffineTransformer():
    '''Class object performs affine transformation
       ___________________________________________
       Example:
       transformer = AffineTransformer(np.random.normal,loc=0,scale=0.0,size=(6))'''
        
    
    @staticmethod
    def distribution(dist_func):
        def generate_dist(**params):
            return dist_func(**params).reshape(3,2)
        return generate_dist

    def __init__(self,dist_func=None, **dist_params):
        self.dist_func = self.distribution(dist_func)
        self.dist_params = dist_params 
        
    def afin_transform(self,image):
        im_shape = image.shape
        
        _ = np.tri(3,2,-1).reshape(-1)
        mask = self.dist_func(**self.dist_params)
        pts1 = np.float32([[im_shape[0] * _[i*2], im_shape[0] * _[i*2+1]]
                         for i in range(3)])
        pts2 = np.float32(pts1 * (mask+1))
        
        M = cv.getAffineTransform(pts1,pts2)
        transformed = cv.warpAffine(image,M,(im_shape[1],im_shape[0]))
        
        return transformed
    
    def __call__(self, image):
        return self.afin_transform(image)
    
