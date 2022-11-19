"""SimilarityServer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from django.urls import path
from app_predict import views

# 根据对应的 url 模式 执行对应的函数
# 这里并不是调用函数，而是指定所要执行的函数
# 这里毕竟本身并不传参
urlpatterns = [
    # path('admin/', admin.site.urls),
    path('predict/', views.predict),
    path('predict_sequence/', views.predict_sequence),
    path('predict_one_sequence/', views.predict_one_sequence),
    path('cos_predict_one_sequence/', views.cos_predict_one_sequence)
]
