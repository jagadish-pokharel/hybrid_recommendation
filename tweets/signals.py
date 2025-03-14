from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from .models import LoginHistory
from django.contrib.auth.models import User
from .models import UserProfile
from django.db.models.signals import post_save



@receiver(user_logged_in)
def log_login(sender, request, user, **kwargs):
    LoginHistory.objects.create(user=user)

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)