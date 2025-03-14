from django.shortcuts import redirect

class GenreSelectionMiddleware:
    """Middleware to ensure users select genres before accessing other pages."""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:  # Check if user is logged in
            if not request.session.get('selected_genres') and request.path not in ["/select-genres/", "/logout/"]:
                return redirect('genre_selection')  # Force user to select genres
        return self.get_response(request)



class NoCacheMiddleware:
    def __init__(self, get_response):  # Correct initialization
        self.get_response = get_response

    def __call__(self, request):  # Correct call method
        response = self.get_response(request) # Call get_response with the request
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        #response['Pragma'] = 'no-cache'  # Optional
        #response['Expires'] = '0'  # Optional
        return response