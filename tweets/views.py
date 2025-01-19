from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from .utily import item_user_matrix,catogery_matrix,aged_matrix,book_data,combined_features
from tweets.functions import get_recommendations_ratings ,get_recommendations_category,get_recommendations_aged,get_book_recommendations_content,combine_recommendations#,is_book_in_dataset# Import the function
from .models import SearchHistory
# Login View

def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        
        if not username or not password:
            messages.error(request, "Both username and password are required!")
            return redirect('login')

       
        user = authenticate(request, username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                
                return redirect('home')
            else:
                messages.error(request, "Your account is inactive. Please contact support.")
                return redirect('login')
        else:
            messages.error(request, "Invalid username or password.")
            return redirect('login')

    return render(request, 'login.html')


# Signup View
def SignupPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        
        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            return redirect('signup')



        
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username is already taken. Please choose another one.")
            return redirect('signup')
        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return redirect('signup')

       
        try:
            user = User.objects.create_user(username=username, email=email, password=password1)
            user.is_active = True  
            user.save()
            messages.success(request, "Account created successfully! You can now log in.")
            return redirect('login')
        except Exception as e:
            messages.error(request, f"An error occurred during signup: {str(e)}")
            return redirect('signup')

    return render(request, 'signup.html')


# Home Page View (requires login)
@login_required(login_url='login')  
def HomePage(request):
    return render(request, 'home.html')


# Logout View
def LogoutPage(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('login')


# def get_recommendations(request):
#     item = request.GET.get('title')
#     recommended_items_ratings = []
#     recommended_items_category = []
#     recommended_items_aged= []
#     recommended_contents=[]
#     if item:
#         # Get recommendations based on the user's search
#         recommended_items_ratings = get_recommendations_ratings(item,item_user_matrix, k=5)  # Assuming this function returns recommendations
#         recommended_items_category = get_recommendations_category(item,catogery_matrix, k=5)
#         recommended_items_aged=get_recommendations_aged(item,aged_matrix,k=5)
#         recommended_contents=get_book_recommendations_content(item,combined_features, book_data,k=5)


#     # Render the recommendations on a separate page (recommendations.html)
#     return render(request, 'recommendations.html', {
#         'recommended_items_ratings': recommended_items_ratings, 
#         'recommended_items_category': recommended_items_category, 
#         'recommended_items_aged':recommended_items_aged,
#         'recommended_contents':recommended_contents,
#         'item': item
#     })
# def get_recommendations(request):
#     item = request.GET.get('title')
#     combined_recommendations = []

#     if item:
#         # Get individual recommendations
#         recommended_items_ratings = get_recommendations_ratings(item, item_user_matrix, k=5) 
#         recommended_items_category = get_recommendations_category(item, catogery_matrix, k=5) 
#         recommended_items_aged = get_recommendations_aged(item, aged_matrix, k=5) 
#         recommended_contents = get_book_recommendations_content(item, combined_features, book_data, k=5) 

#         # Define weights for each type of recommendation
#         weight_ratings = 0.4
#         weight_category = 0.3
#         weight_aged = 0.2
#         weight_content = 0.1

#         # Combine recommendations using weighted average
#         combined_recommendations = combine_recommendations(
#             recommended_items_ratings,
#             recommended_items_category,
#             recommended_items_aged,
#             recommended_contents,
#             weight_ratings,
#             weight_category,
#             weight_aged,
#             weight_content,
#             book_data
#         )

#     context = {
#         'combined_recommendations': combined_recommendations,
#         'item': item
#     }

#     return render(request, 'combined.html', context)
def get_recommendations(request): 
    item = request.GET.get('title')
    combined_recommendations = []

    if item:


        # Check if item exists in each recommendation system
        recommended_items_ratings = get_recommendations_ratings(item, item_user_matrix, k=5)
        print(f"Ratings Recommendations for '{item}': {recommended_items_ratings}")
        
        recommended_items_category = get_recommendations_category(item, catogery_matrix, k=5)
        print(f"Category Recommendations for '{item}': {recommended_items_category}")
        
        recommended_items_aged = get_recommendations_aged(item, aged_matrix, k=5)
        print(f"Aged Recommendations for '{item}': {recommended_items_aged}")
        
        recommended_contents = get_book_recommendations_content(item, combined_features, book_data, k=5)
        print(f"Content Recommendations for '{item}': {recommended_contents}")

        # Define weights for each type of recommendation
        weight_ratings = 0.4
        weight_category = 0.3
        weight_aged = 0.2
        weight_content = 0.1

        # Combine recommendations using weighted average
        combined_recommendations = combine_recommendations(
            recommended_items_ratings,
            recommended_items_category,
            recommended_items_aged,
            recommended_contents,
            weight_ratings,
            weight_category,
            weight_aged,
            weight_content,
            book_data
        )
        
        # Debug output of combined recommendations
        print(f"Combined Recommendations for '{item}': {combined_recommendations}")

        # If no recommendations are available in any system, return a default message
        if not combined_recommendations:
            combined_recommendations = ["No recommendations available."]

    context = {
        'combined_recommendations': combined_recommendations,
        'item': item
    }

    return render(request, 'combined.html', context)




def ratings_page(request):
    return render(request, 'ratings.html')


def ratings_recommendations(request):
    recommendations = []
    book_name = None

    if request.method == 'POST':
        book_name = request.POST.get('bookName')  # Get the book name from the form
        if book_name:
            recommendations = get_recommendations_ratings(book_name,item_user_matrix, k=5) # Generate recommendations
    
    return render(request, 'ratings.html', {
        'book_name': book_name,  # Pass the book name back to the template
        'recommendations': recommendations  # Pass the recommendations to the template
    })


def aged_page(request):
    return render(request, 'aged.html')




def ratings_aged_recommendations(request):
    recommendations = []
    book_name = None

    if request.method == 'POST':
        book_name = request.POST.get('bookName')  # Get the book name from the form
        if book_name:
            # Store the search query in SearchHistory
            SearchHistory.objects.create(user=request.user, search_query=book_name)
            
            # Generate recommendations
            recommendations = get_recommendations_aged(book_name, aged_matrix, k=5)
    
    return render(request, 'aged.html', {
        'book_name': book_name,  # Pass the book name back to the template
        'recommendations': recommendations  # Pass the recommendations to the template
    })



def category_page(request):
    return render(request, 'category.html')


def category_recommendations(request):
    recommendations = []
    book_name = None

    if request.method == 'POST':
        book_name = request.POST.get('bookName')  # Get the book name from the form
        if book_name:
            # Generate recommendations
            recommendations = get_recommendations_category(book_name, aged_matrix, k=5)
    
    return render(request, 'category.html', {
        'book_name': book_name,  # Pass the book name back to the template
        'recommendations': recommendations  # Pass the recommendations to the template
    })


def content_page(request):
    return render(request, 'content.html')

def content_recommendations(request):
    recommendations = []
    book_name = None

    if request.method == 'POST':
        book_name = request.POST.get('bookName')  # Get the book name from the form
        if book_name:
            # Generate recommendations
            recommendations = get_book_recommendations_content(book_name, combined_features, book_data,k=5)
    return render(request, 'content.html', {
        'book_name': book_name,  # Pass the book name back to the template
        'recommendations': recommendations  # Pass the recommendations to the template
    })
