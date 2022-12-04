from django.db import models

# Create your models here.

class dangNhap(models.Model):
    maDangNhap=models.CharField(null=False,max_length=255)
    matKhau=models.CharField(null=False,max_length=255)

class laThuoc(models.Model):
    maLa=models.CharField(null=False,max_length=255)
    tenLa=models.CharField(null=False,max_length=255)
    tenKhac=models.CharField(null=True,max_length=255,blank=True)
    tenKhoaHoc=models.CharField(null=False,max_length=255)
    moTa=models.TextField(null=True,blank=True)
    maDieuTri=models.CharField(null=False,max_length=255)
    giaBan=models.IntegerField(null=False)
    phanBo=models.TextField(null=True,blank=True)
    cachDung=models.TextField(null=True,blank=True)
    soLuongCon=models.IntegerField(null=False)
    hinhAnh=models.ImageField(null=False,upload_to='lathuoc',default=None)

class dieuTri(models.Model):
    maDieuTri=models.CharField(null=False,max_length=255)
    maBenh=models.CharField(null=False,max_length=255)

class tinTuc(models.Model):
    maTinTuc=models.CharField(null=False,max_length=255)
    ngayDang=models.DateTimeField(null=False)
    tieuDe=models.CharField(null=False,max_length=255)

class ctTinTuc(models.Model):
    maTinTuc=models.CharField(null=False,max_length=255)
    noiDung=models.TextField(null=False,blank=False)
    viTri=models.IntegerField(null=False)
    kieuDuLieu=models.CharField(null=False,max_length=255)

class donHang(models.Model):
    maDonHang=models.CharField(null=False,max_length=255)
    maKhachHang=models.CharField(null=False,max_length=255)
    ngayLap=models.DateTimeField(null=False)
    
class ctDonHang(models.Model):
    maDonHang=models.CharField(null=False,max_length=255)
    maLa=models.CharField(null=False,max_length=255)
    soLuong=models.IntegerField(null=False)

class benhGan(models.Model):
    maBenh=models.CharField(null=False,max_length=255)
    timHieuChung=models.TextField(null=False,blank=False)
    nguyenNhan=models.TextField(null=False,blank=False)
    nguyCo=models.TextField(null=False,blank=False)
    dieuTri=models.TextField(null=False,blank=False)
    cheDoSinhHoat=models.TextField(null=False,blank=False)

class clipboard(models.Model):
    file=models.ImageField(upload_to='clipboard',null=False,default=None)