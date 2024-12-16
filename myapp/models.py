from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class HealthInfo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='health_info')
    phone_number = models.CharField(max_length=20, blank=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    heart_rate = models.FloatField()
    low_bp = models.FloatField()
    high_bp = models.FloatField()
    height = models.FloatField()
    weight = models.FloatField()
    body_temperature = models.FloatField()
    prediction_result = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Health Info for {self.user.username}"