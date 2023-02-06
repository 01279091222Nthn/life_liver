from rest_framework import serializers
from .models import *

class LaCaySerializer(serializers.ModelSerializer):
    class Meta:
        model=laThuoc
        fields=('maLa','tenLa','tenKhac','tenKhoaHoc','giaBan','soLuongCon','hinhAnh','noiDungKhac')

class BenhGanSerializer(serializers.ModelSerializer):
    class Meta:
        model=benhGan
        fields=('maBenh','tenBenh','hinhAnh')

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
        fields=('maDonHang','maKhachHang','ngayLap')


class CTDonHangSerializer(serializers.ModelSerializer):
    class Meta:
        model=ctDonHang
        fields=('maCTDonHang','maLa','soLuong')

class DieuTriSerializer(serializers.ModelSerializer):
    class Meta:
        model=dieuTri
        fields=('maLa','maBenh')