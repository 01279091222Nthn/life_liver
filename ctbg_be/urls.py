"""ctbg_be URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from module import views
from rest_framework.routers import DefaultRouter
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions
from django.conf import settings
from django.conf.urls.static import static

router=DefaultRouter()
router.register('lathuoc',views.LaCayViewSet)
router.register('benhgan',views.BenhGanViewSet)
router.register('tintuc',views.TinTucViewSet)
router.register('upload',views.UploadViewSet)
router.register('khachhang',views.KhachHangViewSet)
router.register('donhang',views.DonHangViewSet)

schema_view = get_schema_view(
   openapi.Info(
      title="Snippets API",
      default_version='v1',
      description="Test description",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="contact@snippets.local"),
      license=openapi.License(name="BSD License"),
   ),
   public=True,
   permission_classes=[permissions.AllowAny],
)


urlpatterns = [
    #special api
    path('admin/', admin.site.urls),
    path('predict/', views.GetPredictedResult.as_view()),
    path('lathuoc/search/',views.searchLaThuoc),
    path('dieutri/update/',views.capNhatDieuTri),
    path('auth/sendotp/',views.sendOTP),
    path('auth/verify/',views.verifyOTP),
    path('tintuc/search/<str:input>/',views.searchTinTuc),
    path('dieutri/<str:maLa>/',views.getDieuTri),
    path('ctdonhang/',views.postCTDonHang),
    path('ctdonhang/get/<str:maDonHang>/',views.getCTDonHang),
    path('donhang/delete/<str:maDonHang>/',views.deleteDonHang),
    path('dangnhap/',views.dangNhap),
    # path('cttintuc/matintuc/<str:maTinTuc>/',views.getCtTinTuc),
    # path('lathuoc/mala/<str:mala>/',views.getCtLaThuoc),
    #basic api
    path('',include(router.urls)),
    path('swagger', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)