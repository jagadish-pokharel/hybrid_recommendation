import joblib

item_user_matrix = joblib.load('tweets/models/item_user_matrix.pkl')
category_matrix=joblib.load('tweets/models/category.pkl')
aged_matrix=joblib.load('tweets/models/aged.pkl')
book_data=joblib.load('tweets/models/book_data.pkl')
combined_features=joblib.load('tweets/models/combined_features.pkl')
book_data_admin = joblib.load("tweets/models/final.pkl")

