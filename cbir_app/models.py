from django.db import models

# Create your models here.

class Product(models.Model):
    image_id = models.CharField(max_length=100)
    feature_vector = models.JSONField(null=True)