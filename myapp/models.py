
# Create your models here.
from django.db import models

class Article(models.Model):
    heading = models.CharField(max_length=255)
    embedding = models.TextField()

    def __str__(self):
        return self.heading
