from django.urls import path
from . import views
from django.urls import path, include

urlpatterns = [
    path('', views.SignupPage, name='signup'),
    path('login/', views.LoginPage, name='login'),
    path('home/', views.HomePage, name='home'),
 
    path('logout/', views.LogoutPage, name='logout'),
    path('base_page/', views.base_page, name='base_page'),
    path('header_page/', views.header_page, name='header_page'),
    path('footer_page/', views.footer_page, name='footer_page'),
    path('aged_page/', views.aged_page, name='aged_page'),
    path('category_page/', views.category_page, name='category_page'),
    path('content_page/', views.content_page, name='content_page'),
    path('ratings_page/', views.ratings_page, name='ratings_page'),
    
    path('get_recommendations/',views.get_recommendations, name='get_recommendations'),
    path('ratings_aged_recommendations/', views.ratings_aged_recommendations, name='ratings_aged_recommendations'),
    path('browse_category_recommendations/', views.category_recommendations, name='category_recommendations'),
    path('ratings_recommendations/', views.ratings_recommendations, name='ratings_recommendations'),
    path('content_recommendations/', views.content_recommendations, name='content_recommendations'),
    path('genre-selection/', views.genre_selection, name='genre_selection'),
    path('get-selection/', views.get_selection, name='get_selection'),
    path('search/', views.search_view, name='search_results'), 
    path('book_details/<str:book_title>/', views.book_details_view, name='book_details'), 
    
]
