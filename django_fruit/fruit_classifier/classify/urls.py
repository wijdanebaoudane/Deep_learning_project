from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_fruit, name='classify_fruit'),
]
