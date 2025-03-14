from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout,get_backends
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.views.decorators.cache import never_cache
from .models import UserProfile, Genre
import logging
import joblib
import re
import pandas as pd
from .functions import recommend_books
from .utily import (
    item_user_matrix, category_matrix, aged_matrix, book_data, combined_features
)
from tweets.functions import (
    get_recommendations_ratings, get_recommendations_category,recommend_books,get_recommendations_aged, 
    get_book_recommendations_content, get_recommendations,get_recommendations_for_book,get_search_results,train_model
)
from .models import SearchHistory,UserProfile, Genre
from difflib import SequenceMatcher 


# Load model components correctly
try:
    model_data = joblib.load("tweets/models/genere.pkl")  # Use cosine model name
    tfidf_category, tfidf_summary, combined_matrix, df = train_model(model_data)
except FileNotFoundError:
    print("Model file not found. Please train and save the model first.")
    tfidf_category, tfidf_summary, combined_matrix, df = None, None, None, None
except Exception as e:
    print(f"Model loading failed: {e}")
    tfidf_category, tfidf_summary, combined_matrix, df = None, None, None, None


#Loginpage      

@never_cache
def LoginPage(request):
    if request.user.is_authenticated:  # Check if already logged in
        return redirect('home')  # Redirect to home if logged in

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            request.session['logged_in'] = True
            request.session['current_step'] = 'home' #Set current step
            

            try:
                profile = UserProfile.objects.get(user=user)
                if profile.selected_genres.exists():
                    # Fetch stored recommendations or generate new ones
                    if 'recommendations' not in request.session:
                        genres = profile.selected_genres.values_list('name', flat=True)
                        recommendations = recommend_books(genres, tfidf_category, tfidf_summary,combined_matrix, df)
                        request.session['recommendations'] = recommendations
                    return redirect('home')
                return redirect('genre_selection')
            except UserProfile.DoesNotExist:
                return redirect('genre_selection')
        else:
            messages.error(request, "Invalid username or password.")

    response = render(request, 'login.html')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'  # Discourage caching
    return response



#SIGNUPPAGE

@never_cache
def SignupPage(request):
    if request.user.is_authenticated:
        return redirect('home')

    if 'logged_in' in request.session and request.session['logged_in']:
        return redirect('home')

    if request.method == 'POST':
        username = request.POST.get('username').strip()
        email = request.POST.get('email').strip()
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if not username or not email or not password1 or not password2:
            messages.error(request, "All fields are required!")
            return redirect('signup')

        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            return redirect('signup')

        if len(password1) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            return redirect('signup')

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            messages.error(request, "Invalid email format.")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username is already taken.")
            return redirect('signup')

        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return redirect('signup')

        user = User.objects.create_user(username=username, email=email, password=password1)
        user.save()

        UserProfile.objects.get_or_create(user=user)

        backend = get_backends()[0]
        user.backend = f"{backend.__module__}.{backend.__class__.__name__}"
        login(request, user, backend=user.backend)

        return redirect('genre_selection')

    return render(request, 'signup.html')




@never_cache
def LogoutPage(request):
    logout(request)
    request.session.flush()  # Clears all session data

    response = redirect('login')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response
#genre selection





@never_cache
@login_required(login_url='login')
def genre_selection(request):
    try:
        # Only fetch the existing profile, do NOT create a new one!
        profile = UserProfile.objects.get(user=request.user)

        if request.method == "POST":
            request.session['genres_selected'] = True
            selected_genres = request.POST.getlist("selected_genres")

            if selected_genres:
                genres = Genre.objects.filter(name__in=selected_genres)
                profile.selected_genres.set(genres)

                # Generate recommendations
                recommendations = recommend_books(
                    selected_genres, 
                    tfidf_category, 
                    tfidf_summary, 
                    combined_matrix, 
                    df
                )

                # Store recommendations in session
                request.session["recommendations"] = recommendations
                request.session.modified = True  

                return redirect("home")  

            else:
                messages.error(request, "")

        if profile.selected_genres.exists():
            return redirect("home")

    except UserProfile.DoesNotExist:
        # If profile doesn't exist (which should not happen), create one
        UserProfile.objects.create(user=request.user)
        return redirect("genre_selection")

    except Exception as e:
        print(f"Error in genre selection: {e}")
        messages.error(request, "")

    return render(request, "genre_selection.html", {
        "genres": Genre.objects.all()
    })







#get selection
@never_cache 
@login_required(login_url='login')
def get_selection(request):
    """Handles genre selection and stores recommendations in session."""
    user = request.user  # Get logged-in user

    try:
        profile = UserProfile.objects.get(user=user)  # Fetch UserProfile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=user)  # Create profile if missing

    selected_genres = request.POST.getlist("selected_genres")

    if not selected_genres:
        messages.error(request, "")
        return render(request, "genre_selection.html", {
            "genres": Genre.objects.all(),  
        })

    #  Store selected genres in the database
    genres = Genre.objects.filter(name__in=selected_genres)
    profile.selected_genres.set(genres)
    profile.save()  # Ensure changes are committed to DB

    print(f" Saved genres for {user.username}: {selected_genres}")  # Debugging output

    # Fetch recommendations using selected genres
    recommendations = recommend_books(selected_genres, tfidf_category, tfidf_summary, combined_matrix, df)

    # Convert DataFrame to a list of dictionaries (for session storage)
    if isinstance(recommendations, pd.DataFrame):
        recommendations = recommendations.to_dict(orient="records")
    elif not recommendations:
        recommendations = []  # Ensure it is always a list

    # Debugging Output
    print(" Final Recommendations Stored in Session:", recommendations if recommendations else "No recommendations")

    #  Store recommendations in session
    request.session["recommendations"] = recommendations
    request.session.modified = True  # Force session to save

    # Redirect to homepage
    return redirect("home")  # Ensure correct URL resolution






#homepage

#     return redirect('genre_selection')

@login_required(login_url='login')
def HomePage(request):
    """Renders homepage with recommendations."""
    user = request.user  # Get logged-in user

    try:
        user_profile = UserProfile.objects.get(user=user)  # Get UserProfile
    except UserProfile.DoesNotExist:
        return redirect('genre_selection')  # Redirect if profile is missing

    if user_profile.selected_genres.exists():
        selected_genres = list(user_profile.selected_genres.values_list('name', flat=True))

        #  Fetch book recommendations from session
        recommendations = request.session.get('recommendations', [])

        #  If no recommendations exist, regenerate them
        if not recommendations:
            recommendations = recommend_books(selected_genres, tfidf_category, tfidf_summary,combined_matrix, df)

            if isinstance(recommendations, pd.DataFrame):
                recommendations = recommendations.to_dict(orient="records")
            elif not recommendations:
                recommendations = []

            request.session['recommendations'] = recommendations
            request.session.modified = True

        print(" Displaying recommendations:", recommendations)  # Debugging Output

        #  Fetch admin recommendations (same as user recommendations for now)
        admin_recommendations = recommendations  # Or fetch admin-set books from DB

        return render(request, "homes.html", {
            "recommendations": recommendations,
            "admin_recommendations": admin_recommendations  #  Ensure it's passed
        })

    return redirect('genre_selection')  # Redirect if no genres are selected

def my_protected_view(request):
    response = render(request, 'my_template.html')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


#.......................Search result and book details view.............


book_data = joblib.load('tweets/models/book_data.pkl')
combined_features = joblib.load('tweets/models/combined_features.pkl')
logger = logging.getLogger(__name__)

# Define a set of common stopwords



 # For ranking titles based on string similarity

# Define a set of common stopwords
STOPWORDS = {"a", "an", "the", "and", "or", "but", "for", "nor", "on", "in", "at", "by", "with", "about", "as", "from", "to", "of"}

def similarity_ratio(str1, str2):
    """Return a similarity ratio between two strings (0 to 1)."""
    return SequenceMatcher(None, str1, str2).ratio()

@never_cache 
def search_view(request):
    query = request.GET.get('title')
    search_type = request.GET.get('search_type', 'hybrid')

    if query:
        query_lower = query.lower()

        # 1. Exact Match Check for Full Phrase (exact match for full query)
        exact_match = book_data[book_data['book_title'].str.lower() == query_lower]
        if not exact_match.empty:
            try:
                book_title = exact_match['book_title'].iloc[0]
                return redirect(f'/book_details/{book_title}/?search_type={search_type}')
            except IndexError:
                pass

        # 2. Flexible Search - Match any word in the query (ignoring stopwords)
        query_words = query_lower.split()  # Split query into individual words

        # Remove stopwords from the list of words
        filtered_words = [word for word in query_words if word not in STOPWORDS]

        if not filtered_words:
            return render(request, 'search.html', {'no_results': True, 'item': query, 'search_type': search_type})

        # First, attempt a full match for any phrase containing these words
        phrase_pattern = r'\b' + r'\b.*\b'.join([re.escape(word) for word in filtered_words]) + r'\b'

        # Perform the search for the full phrase with the filtered words
        results = book_data[book_data['book_title'].str.lower().str.contains(phrase_pattern, regex=True)]

        if results.empty:
            # If no results are found, we fall back to individual word matching
            individual_word_pattern = r'\b' + r'\b|\b'.join([re.escape(word) for word in filtered_words]) + r'\b'
            results = book_data[book_data['book_title'].str.lower().str.contains(individual_word_pattern, regex=True)]

        if results.empty:
            return render(request, 'search.html', {'no_results': True, 'item': query, 'search_type': search_type})

        # Rank results based on similarity to the query
        results['similarity'] = results['book_title'].apply(lambda title: similarity_ratio(query_lower, title.lower()))

        # Sort by similarity (highest first)
        ranked_results = results.sort_values(by='similarity', ascending=False)

        # Limit to the first 4 results
        ranked_results = ranked_results[:4]
        results_list = ranked_results.to_dict('records')

        return render(request, 'search.html', {'search_results': results_list, 'item': query, 'search_type': search_type})
    else:
        return render(request, 'search.html')


@never_cache
def book_details_view(request, book_title):
    search_type = request.GET.get('search_type')
    logging.debug(f"Book details view: book_title={book_title}, search_type={search_type}")

    try:
        book = book_data[book_data['book_title'].str.lower() == book_title.lower()].iloc[0].to_dict()
    except IndexError:
        return redirect('search')

    recommendations = get_recommendations_for_book(book_title, combined_features, book_data, search_type, k=10)
    

    # Debugging: Print the type and content of 'recommendations'
    logging.debug(f"Recommendations type: {type(recommendations)}")
    logging.debug(f"Recommendations content: {recommendations}")

    context = {
        'book_title': book_title,
        'book': book,
        'recommendations': recommendations,
        'search_type': search_type,
    }
    return render(request, 'book.html', context)
# ------------------- Recommendation Views -------------------


@never_cache 
def combined_recommendations_view(request):  # This is your main view now
    item = request.GET.get('title')  # Get the search query (title or category)
    search_type = request.GET.get('search_type', 'title')  # Get the search type (default to title)

    combined_recommendations = []  # Initialize an empty list for recommendations

    if item:  # If the user has entered a search query
        if request.user.is_authenticated:  # If the user is logged in
            SearchHistory.objects.create(user=request.user, search_query=item)  # Save search history

        # ***THIS IS THE KEY CHANGE***
        combined_recommendations = get_recommendations(
            item,  # The search query (title or category)
            search_type,  # The search type ("title" or "category")
            item_user_matrix,  # Your item-user matrix
            category_matrix,  # Your category matrix
            aged_matrix,  # Your aged matrix
            combined_features,  # Your combined features matrix
            book_data  # Your book data DataFrame
        )

        if not combined_recommendations:  # If no recommendations were found
            combined_recommendations = [{"book_title": "No recommendations available.", "url": None}]

    context = {
        'combined_recommendations': combined_recommendations,  # Pass recommendations to the template
        'item': item,  # Pass the search query to the template
        'search_type': search_type,  # Pass the search type to the template
    }

    return render(request, 'combined.html', context) 




        

        

# ------------------- Page Rendering Views -------------------
@login_required(login_url='login')
def base_page(request):
    return render(request, 'base.html')

@login_required(login_url='login')
def header_page(request):
    return render(request, 'header.html')


@login_required(login_url='login')
def footer_page(request):
    return render(request, 'footer.html')



def ratings_page(request):
    return render(request, 'ratings.html')



def category_page(request):
    return render(request, 'browse_category.html')



def aged_page(request):
    return render(request, 'aged.html')



def content_page(request):
    return render(request, 'content.html')

# ------------------- Specific Recommendation Views -------------------

def ratings_recommendations(request):
    if request.method == 'POST':
        book_name = request.POST.get('bookName')
        if book_name:
            search_results = get_recommendations_ratings(book_name, item_user_matrix, book_data, k=10)
            return render(request, 'search.html', {'search_results': search_results, 'book_name': book_name, 'recommendation_type': 'ratings'})
    return render(request, 'ratings.html', {})

def category_recommendations(request):
    if request.method == 'POST':
        book_name = request.POST.get('bookName')
        if book_name:
            try:
                recommendations = get_recommendations_category(book_name, category_matrix, book_data, k=10) 
                return render(request, 'browse_category.html', {'book_name': book_name, 'recommendations': recommendations, 'recommendation_type': 'category'})
            except KeyError:
                return render(request, 'browse_category.html', {'error': 'Book not found. Please check the book name.'})
            except Exception as e:
                return render(request, 'browse_category.html', {'error': f'An unexpected error occurred: {e}'})

    return render(request, 'browse_category.html', {})

def ratings_aged_recommendations(request):
    if request.method == 'POST':
        book_name = request.POST.get('bookName')
        if book_name:
            recommendations = get_recommendations_aged(book_name, aged_matrix, book_data, k=10)
            return render(request, 'aged.html', {'book_name': book_name, 'recommendations': recommendations, 'recommendation_type':'aged'})
    return render(request, 'aged.html', {})

def content_recommendations(request):
    if request.method == 'POST':
        book_name = request.POST.get('bookName')
        if book_name:
            recommendations = get_book_recommendations_content(book_name, combined_features, book_data, k=10)
            return render(request, 'content.html', {'book_name': book_name, 'recommendations': recommendations, 'recommendation_type':'content'})
    return render(request, 'content.html', {})