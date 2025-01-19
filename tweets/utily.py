import joblib

knn_ratings = joblib.load('tweets/models/knn_ratings.pkl')
item_user_matrix = joblib.load('tweets/models/item_user_matrix.pkl')
knn_category=joblib.load('tweets/models/knn_category.pkl')
catogery_matrix=joblib.load('tweets/models/category.pkl')
aged_matrix=joblib.load('tweets/models/aged.pkl')
book_data=joblib.load('tweets/models/book_data.pkl')
combined_features=joblib.load('tweets/models/combined_features.pkl')
