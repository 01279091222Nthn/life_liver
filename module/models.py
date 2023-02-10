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
    giaBan=models.IntegerField(null=False)
    soLuongCon=models.IntegerField(null=False)
    hinhAnh=models.ImageField(null=False,default=None,upload_to='uploads')
    noiDungKhac=models.TextField(null=True,blank=True)

class upload(models.Model):
    uploadId = models.CharField(null=False,max_length=255)
    uploadContent = models.CharField(null=False,max_length=255)
    file = models.FileField(null=False,upload_to='uploads')

class dieuTri(models.Model):
    maLa=models.CharField(null=False,max_length=255)
    maBenh=models.CharField(null=False,max_length=255)

class tinTuc(models.Model):
    maTinTuc=models.CharField(null=False,max_length=255)
    ngayDang=models.DateField(null=False)
    tieuDe=models.CharField(null=False,max_length=255)
    hinhAnh=models.ImageField(null=False,default=None,upload_to='uploads')
    noiDungKhac=models.TextField(null=True,blank=True)

class khachHang(models.Model):
    maKhachHang=models.CharField(null=False,max_length=255)
    soDienThoai=models.CharField(null=False,max_length=255)
    tenKhachHang=models.CharField(null=False,max_length=255)
    diaChi=models.TextField(null=False)
    ghiChu=models.TextField(null=True,blank=True)

class donHang(models.Model):
    maDonHang=models.CharField(null=False,max_length=255)
    maKhachHang=models.CharField(null=False,max_length=255)
    ngayLap=models.DateTimeField(null=False)
    tongTien=models.IntegerField(null=False,default=0)
    trangThai=models.IntegerField(null=False,default=0)
    ghiChu=models.TextField(null=True,blank=True)
    
class ctDonHang(models.Model):
    maCTDonHang=models.CharField(null=False,max_length=255)
    maLa=models.CharField(null=False,max_length=255)
    soLuong=models.IntegerField(null=False)

class benhGan(models.Model):
    maBenh=models.CharField(null=False,max_length=255)
    tenBenh=models.CharField(null=False,max_length=255,default=None)
    hinhAnh=models.ImageField(null=False,default=None,upload_to='uploads')
    noiDungKhac=models.TextField(null=True,blank=True)

class clipboard(models.Model):
    file=models.ImageField(upload_to='clipboard',null=False,default=None)

class dataTrain(models.Model):
    label= models.CharField(null=False,max_length=255)
    image=models.ImageField(upload_to='data_train',null=False,default=None)