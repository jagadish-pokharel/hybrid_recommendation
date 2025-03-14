# tweets/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.utils.timezone import now
from django.core.exceptions import ValidationError

# tweets/models.py

class LoginHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    login_time = models.DateTimeField(auto_now_add=True)  # No default
    ip_address = models.GenericIPAddressField(null=True, blank=True)

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.CharField(max_length=255, blank=True)
    search_query = models.TextField()
    search_time = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)  # No default
    
class Genre(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.name


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    selected_genres = models.ManyToManyField(Genre)
    created_at = models.DateTimeField(auto_now_add=True)  # Remove any default=... here

    def __str__(self):
        return f"{self.user.username}'s Profile"






