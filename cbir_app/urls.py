from django.urls import path
from .views import upload_image,upload_image_query

urlpatterns = [
    # Other URL patterns in your project...
    path('', upload_image, name='upload_image'),
    path('upload_image_query/', upload_image_query, name='upload_image_query'),
]
