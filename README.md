## Deep-learning-utilities
 
 **Feed your own data through my scripts and train your own customized neural networks.**
 
 For creating dataset for network use(DatasetCreator.py)[https://github.com/chestnut3108/Deep-learning-utilities/blob/master/DatasetCreator.py].
 
 Your training data must be formmated like this:  
 '''

  data_dir name/class1/imgs*      
                class2/imgs*      
                class3/imgs*      
                .    
                .      
                classn/imgs*      

'''

 
  _DatasetCreator.TrainingDataCreator_ function is used for creating dataset, it returns a single array of all the images merged and preprocessed into one along with another array of corresponding labels.To use this utility just pass the path to your training image directory.
 
