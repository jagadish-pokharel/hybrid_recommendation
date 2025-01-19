# accounts/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.SignupPage, name='signup'), 
    path('login/', views.LoginPage, name='login'),
    path('home/', views.HomePage, name='home'),
    path('logout/', views.LogoutPage, name='logout'),
    path('get_recommendations/',views.get_recommendations, name='get_recommendations'), 
    path('ratings/', views.ratings_page, name='ratings_page'),
    path('ratings_recommendations/', views.ratings_recommendations, name='ratings_recommendations'),
    path('aged/', views.aged_page, name='aged_page'),
    path('ratings_aged_recommendations/', views.ratings_aged_recommendations, name='ratings_aged_recommendations'),
    path('category/', views.category_page, name='category_page'),
    path('category_recommendations/', views.category_recommendations, name='category_recommendations'),
    path('content/', views.content_page, name='content_page'),
    path('content_recommendations/', views.content_recommendations, name='content_recommendations'),

]
