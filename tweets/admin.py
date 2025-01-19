from django.contrib import admin
from django.contrib import admin
from .models import LoginHistory, SearchHistory# Book, BookRating

@admin.register(LoginHistory)
class LoginHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'login_time')
    list_filter = ('login_time',)
    search_fields = ('user__username',)
    ordering = ('-login_time',)   # Specify fields to display in list view


@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'search_query', 'search_time')  # Specify fields to display
    list_filter = ('search_time',)  
    search_fields = ('user__username', 'search_query')  
    ordering = ('-search_time',)


#admin.site.register(Book)
#admin.site.register(BookRating)

