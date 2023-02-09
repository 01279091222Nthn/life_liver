import os
from rest_framework import status, viewsets
from rest_framework import generics
from django.http import JsonResponse
from rest_framework.views import APIView, Response
from django.db.models.signals import pre_delete
from django.dispatch.dispatcher import receiver
from rest_framework.decorators import api_view
from twilio.rest import Client
from django.db import connection

# import train model
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#import models
from .models import *
from .models import dangNhap as modelDangNhap
from .serializers import *

# thư viện sử lý predict
from keras.models import load_model
import numpy as np
import cv2

# Create your views here.


class GetPredictedResult(APIView):
    resnet_model = load_model('/Users/haidang/Downloads/resnet50_take5.model')
    class_names = ['An Xoa', 'Cà Gai Leo', 'Mã Đề', 'Sam Biển', 'Dây Thìa Canh', 'Đu Đủ',
                   'Lá Dâu Tầm', 'Lá Ô Liu', 'Lá Sen', 'Ngải Tía', 'Nghệ Xanh', 'Ngô', 'Trái Mấm', 'Xạ Đen']

    def post(self, request):
        # delete old images
        clipboard.objects.all().delete()
        # post new images
        images = request.FILES.getlist('file')
        for image in images:
            clipboard.objects.create(file=image)
        # predict
        folder = './media/clipboard'
        list = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            image_resized = cv2.resize(image, (224, 224))
            image = np.expand_dims(image_resized, axis=0)
            pred = self.resnet_model.predict(image)
            accuracy = float("{:.2f}".format(max(pred[0]))) * 100
            accuracy = int(accuracy)
            result = {"type": None, "acc": None, "image": None}
            result['type'] = self.class_names[np.argmax(pred)]
            result['acc'] = accuracy
            result['image'] = 'http://127.0.0.1:8000/media/clipboard/'+filename
            list.append(result)
        return JsonResponse(list, status=status.HTTP_201_CREATED, safe=False, json_dumps_params={'ensure_ascii': False})

@api_view(['POST'])
def postDataTrain(request):
    folder_name = request.POST.get('label')
    images = request.FILES.getlist('image')
    BASE_DIR = './media/data_train'
    folder_path = os.path.join(BASE_DIR, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for image in images:
        dataTrain.objects.create(label=folder_name, image=image)



class LaCayViewSet(viewsets.ViewSet, generics.ListCreateAPIView, generics.RetrieveUpdateDestroyAPIView):
    queryset = laThuoc.objects.all()
    serializer_class = LaCaySerializer
    lookup_field = 'maLa'


class BenhGanViewSet(viewsets.ModelViewSet):
    queryset = benhGan.objects.all()
    serializer_class = BenhGanSerializer
    lookup_field = 'maBenh'


class TinTucViewSet(viewsets.ModelViewSet):
    queryset = tinTuc.objects.all()
    serializer_class = TinTucSerializer
    lookup_field = 'maTinTuc'


class UploadViewSet(viewsets.ModelViewSet):
    queryset = upload.objects.all()
    serializer_class = UploadSerializer


class DonHangViewSet(viewsets.ViewSet,generics.ListCreateAPIView,generics.UpdateAPIView):
    queryset = donHang.objects.all()
    serializer_class = DonHangSerializer
    lookup_field = 'maDonHang'

class KhachHangViewSet(viewsets.ViewSet,generics.CreateAPIView):
    queryset = khachHang.objects.all()
    serializer_class = KhachHangSerializer
    lookup_field ='maKhachHang'

@api_view(['POST'])
def dangNhap(request):
    maDangNhap = request.POST.get('maDangNhap')
    matKhau = request.POST.get('matKhau')
    if modelDangNhap.objects.filter(maDangNhap=maDangNhap,matKhau=matKhau).exists():
        return Response(status=status.HTTP_200_OK)
    else:
        return Response(status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
def capNhatDieuTri(request):
    mala = request.POST.get('maLa')
    mabenh = request.POST.getlist('maBenh[]')
    dieuTri.objects.filter(maLa=mala).delete()
    for mb in mabenh:
        dieuTri.objects.create(maLa=mala, maBenh=mb)
    return Response(status=status.HTTP_200_OK)


@api_view(['POST'])
def searchLaThuoc(request):
    tenLa = request.POST.get('tenLa')
    maBenh = request.POST.getlist("maBenh[]")
    lathuoc = ''
    if len(tenLa) == 0 and len(maBenh) > 0:
        lathuoc = laThuoc.objects.raw(
            """
            SELECT module_lathuoc.*
            FROM module_lathuoc, module_dieutri
            WHERE module_lathuoc.maLa=module_dieutri.maLa
            AND module_dieutri.maBenh in %s
            GROUP BY module_lathuoc.maLa
            """, [maBenh])

    elif len(tenLa) > 0 and len(maBenh) == 0:
        lathuoc = laThuoc.objects.raw(
            """
            SELECT module_lathuoc.*
            FROM module_lathuoc
            WHERE module_lathuoc.tenLa LIKE %s
            """, [f'%{tenLa}%'])

    elif len(tenLa) > 0 and len(maBenh) > 0:
        lathuoc = laThuoc.objects.raw(
            """
            SELECT module_lathuoc.*
            FROM module_lathuoc, module_dieutri
            WHERE module_lathuoc.maLa=module_dieutri.maLa
            AND module_lathuoc.tenLa LIKE %s
            AND module_dieutri.maBenh in %s
            GROUP BY module_lathuoc.maLa
            """, [f'%{tenLa}%', maBenh])

    else:
        lathuoc = laThuoc.objects.all()

    lathuoc_serializer = LaCaySerializer(
        lathuoc, many=True, context={"request": request})
    return Response(lathuoc_serializer.data, status=status.HTTP_200_OK)


@api_view(['GET'])
def searchTinTuc(request, input):
    tintuc = tinTuc.objects.filter(tieuDe__icontains=input).order_by('ngayDang')
    tintuc_serializer = TinTucSerializer(
        tintuc, many=True, context={"request": request})
    return Response(tintuc_serializer.data, status.HTTP_200_OK)


@api_view(['GET'])
def getDieuTri(request, maLa):
    dieutri = dieuTri.objects.filter(maLa=maLa)
    dieuTriSer = DieuTriSerializer(dieutri, many=True)
    return Response(dieuTriSer.data, status=status.HTTP_200_OK)




# Set environment variables for your credentials
# Read more at http://twil.io/secure

@api_view(['POST'])
def sendOTP(request):
    number = request.POST.get('number')
    account_sid = "ACc6abd338c8a37ba8e41c9084486dffb5"
    auth_token = "5207ee86233e7cfb2d862aea01e5e308"
    verify_sid = "VA38abf9e9316fe448055603fb24135402"
    client = Client(account_sid, auth_token)
    client.verify.services(verify_sid).verifications.create(
        to=f"+84815734366",
        channel="sms"
    )
    return Response(status=status.HTTP_200_OK)


@api_view(['POST'])
def verifyOTP(request):
    otp = request.POST.get('otp')
    account_sid = "ACc6abd338c8a37ba8e41c9084486dffb5"
    auth_token = "5207ee86233e7cfb2d862aea01e5e308"
    verify_sid = "VA38abf9e9316fe448055603fb24135402"
    client = Client(account_sid, auth_token)
    print(otp)
    client.verify.services(verify_sid).verification_checks.create(
        to="+84815734366",
        code=otp
    )
    return Response(status=status.HTTP_200_OK)

@api_view(['POST'])
def postCTDonHang(request):
    maDon = request.POST.get('maDonHang')
    sls = request.POST.getlist('soLuong[]')
    maLas = request.POST.getlist('maLa[]')
    i=0
    for maLa in maLas:
        cursor=connection.cursor()
        cursor.execute(
            """
            UPDATE module_lathuoc SET soLuongCon=soLuongCon-%s WHERE maLa=%s
            """
        ,[sls[i],maLa])
        
        ctDonHang.objects.create(maCTDonHang=maDon, maLa=maLa, soLuong=sls[i])
        i+=1
    return Response(status=status.HTTP_200_OK)
    
@api_view(['GET'])
def getCTDonHang(request,maDonHang):
    dh=donHang.objects.get(maDonHang=maDonHang)
    dh= DonHangSerializer(dh)
    ctdh=ctDonHang.objects.filter(maCTDonHang=maDonHang)
    ctdh= CTDonHangSerializer(ctdh,many=True,context={"request": request})
    kh=khachHang.objects.get(maKhachHang=maDonHang)
    kh=KhachHangSerializer(kh)
    donhang={
        'DonHang':dh.data,
        'CTDonHang':ctdh.data,
        'KhachHang':kh.data,
    }
    return Response(donhang,status=status.HTTP_200_OK)

@api_view(['DELETE'])
def deleteDonHang(request, maDonHang):
    donHang.objects.filter(maDonHang=maDonHang).delete()
    ctDonHang.objects.filter(maCTDonHang=maDonHang).delete()
    khachHang.objects.filter(maKhachHang=maDonHang).delete()
    cursor=connection.cursor()
    cursor.execute(
        """
            UPDATE module_lathuoc 
            SET module_lathuoc.soLuongCon=module_lathuoc.soLuongCon+(SELECT module_ctdonhang.soLuong FROM module_lathuoc, module_ctdonhang WHERE module_lathuoc.maLa = module_ctdonhang.maLa AND module_ctdonhang.maCTDonHang=%s)
            WHERE module_lathuoc.maLa=(SELECT module_lathuoc.maLa FROM module_lathuoc, module_ctdonhang WHERE module_lathuoc.maLa = module_ctdonhang.maLa AND module_ctdonhang.maCTDonHang=%s);
        """
    ,[maDonHang,maDonHang])
    return Response(status=status.HTTP_200_OK)

# signals
@receiver(pre_delete, sender=clipboard)
def mymodel_delete(sender, instance, **kwargs):
    instance.file.delete(False)


@receiver(pre_delete, sender=upload)
def mymodel_delete(sender, instance, **kwargs):
    instance.file.delete(False)
