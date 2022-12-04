from rest_framework import serializers
from .models import *

class LaCaySerializer(serializers.ModelSerializer):
    class Meta:
        model=laThuoc
        fields=('maLa','tenLa','tenKhac','tenKhoaHoc','moTa','maDieuTri','giaBan','phanBo','cachDung','soLuongCon','hinhAnh')

class BenhGanSerializer(serializers.ModelSerializer):
    class Meta:
        model=benhGan
        fields=('maBenh','timHieuChung','nguyenNhan','nguyCo','dieuTri','cheDoSinhHoat')

class ClipboardSerializer(serializers.ModelSerializer):
    class Meta:
        model=clipboard
        fields=('file',)

class tintuc (serializers.ModelSerializer):
    class Meta:
        model=tinTuc
        fields=('matintuc','ngayDang','tieuDe','nguyCo','dieuTri','cheDoSinhHoat')