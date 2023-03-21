from django.db import models
from PIL import Image

class IMG(models.Model):
	img = models.ImageField(upload_to='upload')

def __str__(self):
    return self.image.path
# Create your models here.
