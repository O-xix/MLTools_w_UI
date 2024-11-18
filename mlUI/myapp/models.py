from django.db import models

class MLModelConfig(models.Model):
    model_name = models.CharField(max_length=100)
    parameter = models.CharField(max_length=100)
    value = models.FloatField()

class DataFile(models.Model):
    file = models.FileField(upload_to='uploads/')
