from io import BytesIO
import os
from django.shortcuts import render
from rest_framework import status,viewsets
from django.http import JsonResponse
from rest_framework.views import APIView
from django.db.models.signals import pre_delete
from django.dispatch.dispatcher import receiver
import json

#import models
from .models import laThuoc
from .serializers import *

#thư viện sử lý predict
from keras.models import load_model
import numpy as np
import cv2

# Create your views here.



class GetPredictedResult(APIView):
    resnet_model=load_model('/Users/haidang/Downloads/resnet50.model')
    class_names=['An Xoa', 'Cà Gai Leo', 'Mã Đề', 'Sam Biển', 'Dây Thìa Canh', 'Đu Đủ', 'Lá Dâu Tầm', 'Lá Ô Liu', 'Lá Sen', 'Ngải Tía', 'Nghệ Xanh', 'Ngô', 'Trái Mấm', 'Xạ Đen']
    def get(self,request):
        folder='./media/clipboard'
        list=[]
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))                              
            image=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            image_resized = cv2.resize(image,(224,224))
            image=np.expand_dims(image_resized,axis=0)
            pred = self.resnet_model.predict(image)
            accuracy = float("{:.2f}".format(max(pred[0]))) * 100
            accuracy = int(accuracy)
            result={"loai":None,"acc":None}
            result['loai']=self.class_names[np.argmax(pred)]
            result['acc']=accuracy
            list.append(result)
            
        return JsonResponse(list, status=status.HTTP_201_CREATED,safe=False, json_dumps_params={'ensure_ascii': False})


class LaCayViewSet(viewsets.ModelViewSet):
    queryset=laThuoc.objects.all()
    serializer_class =LaCaySerializer


class BenhGanViewSet(viewsets.ModelViewSet):
    queryset=benhGan.objects.all()
    serializer_class =BenhGanSerializer

class ClipBoardViewSet(viewsets.ModelViewSet):
    queryset=clipboard.objects.all()
    serializer_class =ClipboardSerializer

#signals
@receiver(pre_delete, sender=clipboard)
def mymodel_delete(sender, instance, **kwargs):
    instance.file.delete(False)

@receiver(pre_delete, sender=laThuoc)
def mymodel_delete(sender, instance, **kwargs):
    instance.hinhAnh.delete(False)
    