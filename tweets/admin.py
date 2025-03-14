# tweets/admin.py
from django.contrib import admin
from .models import (
    SearchHistory, 
    LoginHistory,
    UserProfile,
    Genre
)

@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'query', 'created_at','search_query', 'search_time']
    list_filter = ('search_time',)  
    search_fields = ('user__username', 'search_query')  
    ordering = ('-search_time',)

@admin.register(LoginHistory)
class LoginHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'login_time', 'ip_address']
    list_filter = ('login_time',)
    search_fields = ('user__username',)
    ordering = ('-login_time',)

# Register other models if needed
admin.site.register(UserProfile)
admin.site.register(Genre)