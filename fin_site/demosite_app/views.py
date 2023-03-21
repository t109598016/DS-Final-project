#from django.shortcuts import render

# Create your views here.


from django.shortcuts import render
from django.http import HttpResponse
from keras.utils.generic_utils import default
import requests
import urllib
from demosite_app.models import IMG
from django import forms
from django.conf import settings
import os
import json
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image as pil_image
import io
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
# tensorflow內定的graph物件 缺少這，後面model.predict()會出錯
import tensorflow as tf

url = "http://localhost:8000/upload/"


model = load_model('fish_VGG16_model.h5')

class UserUploadForm(forms.Form):
	img_url = forms.URLField(label='輸入照片連結', required = True, widget=forms.TextInput(attrs={'size': '60'}))

def image_classify_api_inside(  payload  ):
	img=pil_image.open(io.BytesIO(payload["img_binary_file"]))
	x = prepare_image( img ) #處理格式(1,224,224,3)
	print(x)
	result = classify_image(x) #預測 得到結果
	print(result)
	#result_json = json.dumps(result)
	return result

@csrf_exempt
def image_classify_api(  request  ):
     if request.method == 'POST':
        img=request.FILES['img_binary_file'].read()
        img=pil_image.open(io.BytesIO(img))
        x = prepare_image( img ) #處理格式(1,224,224,3)
        result = classify_image(x) #預測 得到結果
        print(result)
        result_json = json.dumps(result)
        return HttpResponse(result_json,content_type="application/json")

       # return HttpResponse( result, content_type="application/json")
     return HttpResponse('沒有提供Get功能，請用Post!') #for GET
 #處理圖形 這裡進來的是一張原始尺寸的數字化之後的圖
def prepare_image(img, target=(224, 224)):
    # 變更圖形的尺寸
    #img = image.load_img(img_path, target_size= target )
    img = img.resize(target) #圖形大小(224,224,3)
    img = image.img_to_array(img) #呼叫keras的image轉成array
    img = np.expand_dims(img, axis=0) #圖形格式(1,224,224,3)
    img = preprocess_input(img) #尺度化為RestNet50的範圍
    # 回傳 已經處理好的數字圖檔
    return img

def classify_image(img):
	y = tf.compat.v1.get_default_graph()
	y = model.predict(img)
	print(y)
	maxcase=list(y[0]).index(max(y[0]))
	Y_train = pd.read_csv('./Y_train.txt', sep=" ", header=None)
	Y_train = Y_train.values.reshape(-1)
	encoder = LabelEncoder()
	encoder.fit(Y_train)
	encoded_Y_train = encoder.transform(Y_train)
	leb= [None]*76
	for i in range(76):
		for n in range(len(encoded_Y_train)):
			if encoded_Y_train[n] == i:
				leb[i]=Y_train[n]
    #print(y_label)
	pred = {} #輸出dictionary
	pred["predictions"] = [] #用list存答案
	answer = {"label": leb[maxcase], "proba": float(y[0][maxcase])}
	pred["predictions"].append(answer)
	return pred

def img_app(  request  ):
	
	if request.method == 'POST':
		form = UserUploadForm(request.POST)
		if form.is_valid():
			#img_url = request.POST['img_url']
			img_url = form.cleaned_data['img_url']
			#img_url = 'http://pic.baike.soso.com/p/20130703/20130703235000-2083895917.jpg'
			print(img_url)
			#img_url =''
			image_path = 'target_image.jpg'
			urllib.request.urlretrieve(img_url, image_path)
			# 酬載 (payload)
			payload = {"file": open(image_path, "rb")}
			result = requests.post(url, files=payload)
			label = 1
			proba = 1
			#輸出格式整理
			result = result.json()['predictions']
			
			firstline = result[0]

			label = firstline['label']
			proba = firstline['proba']
			proba = "{0:.0f}".format( proba* 100)
			print(label, proba)
		
	else:
		result=''
		label=''
		proba=''
		img_url=""
		form = UserUploadForm()

	return render(request, 'showResult.html',{'form':form, 'label': label, 'proba': proba,'img_url':img_url},)

def uploadImg(request):
	if request.method == 'POST':
		new_img = IMG(
		img=request.FILES.get('image')
		)
		imgg =request.FILES['image'].name
		new_img.save()
		#img_url='C:/Users/Eddie/Desktop/0109/django_img_classify_callAPI/C:/Users/Acer-Travelmate/Documents/django_img_classify_callAPI/media/upload/'+imgg
		image = open('./upload/'+imgg,"rb").read()
		# 酬載 (payload)
		payload = {"img_binary_file": image}
		#result = requests.post(url, files=payload)
		firstline=image_classify_api_inside(payload)
		#輸出格式整理
		firstline = firstline['predictions']
		firstline = firstline[0]
		label = firstline['label']
		proba = firstline['proba']
		proba = "{0:.0f}".format( proba* 100)
		print(label, proba)
		
	else:
		result=''
		label=''
		proba=''
		img_url=""
	
	return render(request, 'photo.html',{'label': label, 'proba': proba},)
