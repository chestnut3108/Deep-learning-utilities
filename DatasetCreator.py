from numpy import array,expand_dims,rollaxis,argmax
from cv2 import cvtColor, imread, COLOR_BGR2GRAY,resize
from os import listdir
from keras.models import load_model
from keras.utils import np_utils
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import h5py
from pandas import DataFrame

def TrainingDataCreator(Path,return_labels=False,image_channels=3,image_dimensions=(150,150),image_dimensions_3 = (150,150,3),backend='tf'):
	'''A deep learning based utility to create dataset from customised dataset 
	Give path to training directory with specifications like tensorflow or theano 
	this utility also preprocess the data

	'''
	#loading all the images with corresponding classes name

	data_dir_list = listdir(Path)

	img_data_list=[]
	training_labels_list=[]
	for dataset in tqdm(data_dir_list):
		img_list=listdir(Path+'/'+ dataset)
		print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
		for img in (img_list):
			input_img=imread(Path + '/'+ dataset + '/'+ img )
			if image_channels==1:
				input_img=cvtColor(input_img, COLOR_BGR2GRAY)
				input_img_resize=resize(input_img,image_dimensions)
			else:
				input_img_resize = resize(input_img,image_dimensions)

			img_data_list.append(input_img_resize)
			training_labels_list.append(dataset)
	

	#basic preprocessing
	img_data = array(img_data_list)
	img_data = img_data.astype('float32')
	img_data /= 255
	

	#image processing according to backend whether tensorflow or theano
	if image_channels==1:
		if backend=='th':
			img_data= expand_dims(img_data, axis=1) 
			print (img_data.shape)
		else:
			img_data= expand_dims(img_data, axis=4) 
			print (img_data.shape)
	else:
		if backend=='th':
			img_data=rollaxis(img_data,3,1)
			print (img_data.shape)



	#preprocessing labels Encoding the labels

	Encoder = LabelEncoder()
	labels_to_categorical_numbers = Encoder.fit_transform(training_labels_list)
	training_labels_encoded = np_utils.to_categorical(labels_to_categorical_numbers)


	#return training data with labels
	if return_labels:
		return img_data, training_labels_encoded, labels_to_categorical_numbers, training_labels_list

	else:
		return img_data,training_labels_encoded




def TestDataCreator(Path,image_channels=3,image_dimensions=(256,256),image_dimensions_3 = (256,256,3),preprocessing=True,backend='tf'):
	'''This is a utility for creating test data
		Todo 1. Add further preprocessing
			 2. Provide mechanism for storing in h5 file formats
			 3. Optimize it further
			 4. Generalize it	
	'''
	img_list = listdir(Path)

	img_data_list=[]
	for img in tqdm(img_list):
		input_img=imread(Path + '/'+ img )
		if image_channels==1:
			input_img=cvtColor(input_img, COLOR_BGR2GRAY)
			input_img_resize=resize(input_img,image_dimensions)
		else:
			input_img_resize = resize(input_img,image_dimensions)
			img_data_list.append(input_img_resize)

	img_data = array(img_data_list)
	img_data = img_data.astype('float32')
	img_data /= 255
	

	#image processing according to backend whether tensorflow or theano
	if image_channels==1:
		if backend=='th':
			img_data= expand_dims(img_data, axis=1) 
			print (img_data.shape)
		else:
			img_data= expand_dims(img_data, axis=4) 
			print (img_data.shape)
	else:
		if backend=='th':
			img_data=rollaxis(img_data,3,1)
			print (img_data.shape)

	return img_data


def mapper(categories,numbers):
	'''another utility function for mapping between encoded data tot classes'''
	combined = []
	
	for i in range(len(categories)):
		combined.append((numbers[i],categories[i]))

	combine = list(set(combined))

	return combine

def TestingScript(model,categories,numbers,image_dimensions=(256,256),Path='./dataset/test_img/'):
	'''This is testing script designed specifically for hackerearth deep learning challenge
		todo #1 Generalize it
			 #2 Reduce dependencies
			 #3 Thats it for now
	''' 

	img_list = listdir(Path)
	output = []
	combine = mapper(categories,numbers)
	for img in tqdm(img_list):
		input_img=imread(Path + '/'+ img )
		input_img_resize = resize(input_img,image_dimensions)
		img_data = array(input_img_resize)
		img_data = img_data.astype('float32')
		img_data /= 255
		img_data= expand_dims(img_data, axis=0) 
		a = model.predict(img_data)
		y_classes = argmax(a,axis=-1)

		for category in range(len(combine)):
			if combine[category][0] == y_classes:
				output.append((img,combine[category][1]))
	final_output = []
	for a in range(len(output)):
		final_output.append((output[a][0][:-4],output[a][1]))

	return final_output


def list_to_csv(final_output,file_name = 'output.csv'):
	''' Very useful utility for converting lists to csv's through pandas data framework
		Can be optimized very much.
	'''
	df = DataFrame(final_output)
	df.to_csv(file_name)
	


def just_predictions(model,categories,numbers,image_dimensions=(256,256),Path='./dataset/test_img/'):
	'''This is testing script designed specifically for hackerearth deep learning challenge
		todo #1 Generalize it
			 #2 Reduce dependencies
			 #3 Thats it for now
	''' 

	img_list = listdir(Path)
	output = []
	for img in tqdm(img_list):
		input_img=imread(Path + '/'+ img )
		input_img_resize = resize(input_img,image_dimensions)
		img_data = array(input_img_resize)
		img_data = img_data.astype('float32')
		img_data /= 255
		img_data= expand_dims(img_data, axis=0) 
		a = model.predict(img_data)
		output.append(a)

	return output
'''
model = load_model('./checkpoints/sept8_1022.h5')

_,_,numbers,categories = TrainingDataCreator(Path='./training',return_labels=True,image_channels=3)

final_output = TestingScript(model,categories,numbers,image_dimensions=(256,256),Path='./dataset/test_img/')
list_to_csv(final_output)
#_,_,numbers,categories = TrainingDataCreator(Path = './training')

h5f = h5py.File('testing_data1.h5', 'w')
h5f.create_dataset('Imgs', data=data)
h5f.create_dataset('Labels',)
h5f.close()'''
