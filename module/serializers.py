from rest_framework import serializers
from .models import *

class LaCaySerializer(serializers.ModelSerializer):
    class Meta:
        model=laThuoc
        fields=('maLa','tenLa','tenKhac','tenKhoaHoc','giaBan','soLuongCon','hinhAnh','noiDungKhac')

class BenhGanSerializer(serializers.ModelSerializer):
    class Meta:
        model=benhGan
        fields=('__all__')

class ClipboardSerializer(serializers.ModelSerializer):
    class Meta:
        model=clipboard
        fields=('file',)

class UploadSerializer(serializers.ModelSerializer):
    class Meta:
        model=upload
        fields=('uploadId','uploadContent','file')

class TinTucSerializer(serializers.ModelSerializer):
    class Meta:
        model=tinTuc
        fields=('maTinTuc','ngayDang','tieuDe','noiDungKhac','hinhAnh')

class DonHangSerializer(serializers.ModelSerializer):
    class Meta:
        model=donHang
        fields=('__all__')

class KhachHangSerializer(serializers.ModelSerializer):
    class Meta:
        model=khachHang
        fields = ('__all__') 

class CTDonHangSerializer(serializers.ModelSerializer):
    class Meta:
        model=ctDonHang
        fields=('__all__')

class DieuTriSerializer(serializers.ModelSerializer):
    class Meta:
        model=dieuTri
        fields=('maLa','maBenh')

class DangNhapSerializer(serializers.ModelSerializer):
    class Meta:
        model=dangNhap
        fields=('__all__')