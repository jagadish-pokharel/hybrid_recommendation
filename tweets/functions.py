import pandas as pd
import numpy as np
from scipy.sparse import issparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse, csr_matrix
import logging
from collections import OrderedDict

from .utily import item_user_matrix,book_data_admin,category_matrix,aged_matrix,combined_features,book_data

from fuzzywuzzy import process  # Import fuzzywuzzy
# def get_recommendations_ratings(item,k):
    

    
#     # Check if item is in the DataFrame index
#     if item not in item_user_matrix.index:
#         return [f"Item '{item}' is not in the dataset."]

#     # Find the index of the item
#     item_index = item_user_matrix.index.get_loc(item)

#     # Compute nearest neighbors
#     distances, indices = knn_ratings.kneighbors(
#         item_user_matrix.iloc[item_index, :].values.reshape(1, -1), n_neighbors=k + 1
#     )

#     # Collect recommendations (excluding the input item itself)
#     recommendations = []
#     for i in range(1, len(distances.flatten())):
#         recommendations.append(item_user_matrix.index[indices.flatten()[i]])

#     return recommendations



############  RATINGS #################

# Custom function to calculate cosine similarity
def get_recommendations_ratings(item, item_user_matrix, book_data, k, url_column='img_l'):
    """Get recommendations based on cosine similarity of ratings."""

    if item not in item_user_matrix.index: #Check if the item exists in the original dataframe index
        return [f"Item '{item}' is not in the dataset."]

    original_index = item_user_matrix.index #Store the original index
    item_index = item_user_matrix.index.get_loc(item)
    target_vector = item_user_matrix.iloc[item_index, :].values.reshape(1, -1)

    if not issparse(item_user_matrix):
        item_user_matrix_sparse = csr_matrix(item_user_matrix)  # Create a *separate* sparse matrix
    else:
        item_user_matrix_sparse = item_user_matrix #If already sparse, use it
    if not issparse(target_vector):
        target_vector = csr_matrix(target_vector)

    similarities = cosine_similarity(target_vector, item_user_matrix_sparse)
    top_indices = similarities[0].argsort()[::-1][1:k + 1]

    recommendations = []
    for i in top_indices:
        book_title = original_index[i]  # Use original_index here!
        book_info = book_data[book_data['book_title'] == book_title]
        url = book_info[url_column].values[0] if not book_info.empty else None
        recommendations.append({'book_title': book_title, 'url': url})

    return recommendations


def get_recommendations_category(item, category_matrix, book_data, k, url_column='img_l'):
    """Get recommendations based on cosine similarity of categories."""

    if item not in category_matrix.index:
        return [f"Item '{item}' is not in the dataset."]

    original_index = category_matrix.index  # Store the original index
    item_index = category_matrix.index.get_loc(item)
    target_vector = category_matrix.iloc[item_index, :].values.reshape(1, -1)

    if not issparse(category_matrix):
        category_matrix_sparse = csr_matrix(category_matrix)  # Create a *separate* sparse matrix
    else:
        category_matrix_sparse = category_matrix  # If already sparse, use it
    if not issparse(target_vector):
        target_vector = csr_matrix(target_vector)

    similarities = cosine_similarity(target_vector, category_matrix_sparse)
    top_indices = similarities[0].argsort()[::-1][1:k + 1]

    recommendations = []
    for i in top_indices:
        book_title = original_index[i]  # Use original_index here!
        book_info = book_data[book_data['book_title'] == book_title]
        if not book_info.empty:
            url = book_info[url_column].values[0]
        else:
            url = None
        recommendations.append({'book_title': book_title, 'url': url})

    return recommendations

###   AGED ####


def get_recommendations_aged(item, aged_matrix, book_data, k, url_column='img_l'):
    """Get recommendations based on cosine similarity of aged data."""

    if item not in aged_matrix.index:
        return [f"Item '{item}' is not in the dataset."]

    original_index = aged_matrix.index  # Store the original index
    item_index = aged_matrix.index.get_loc(item)
    target_vector = aged_matrix.iloc[item_index, :].values.reshape(1, -1)

    if not issparse(aged_matrix):
        aged_matrix_sparse = csr_matrix(aged_matrix)  # Create a *separate* sparse matrix
    else:
        aged_matrix_sparse = aged_matrix #If already sparse, use it
    if not issparse(target_vector):
        target_vector = csr_matrix(target_vector)

    similarities = cosine_similarity(target_vector, aged_matrix_sparse)
    top_indices = similarities[0].argsort()[::-1][1:k + 1]

    recommendations = []
    for i in top_indices:
        book_title = original_index[i]  # Use original_index here!
        book_info = book_data[book_data['book_title'] == book_title]
        url = book_info[url_column].values[0] if not book_info.empty else None
        recommendations.append({'book_title': book_title, 'url': url})

    return recommendations



#### import BOOK title combined_features and book_data pickle jasari aagi ko jasto




def get_book_recommendations_content(book_title, combined_features, book_data, k, url_column='img_l'):
    """Get recommendations based on cosine similarity of combined features."""

    if book_title not in book_data['book_title'].values:
        return f"Book title '{book_title}' is not in the dataset."

    original_index = book_data.index  # Store original index of book_data
    target_index = book_data[book_data['book_title'] == book_title].index[0]
    
    if issparse(combined_features):
        target_vector = combined_features[target_index].toarray().flatten().reshape(1, -1)
    else:
        target_vector = combined_features[target_index].reshape(1, -1)

    if not issparse(combined_features):
        combined_features_sparse = csr_matrix(combined_features)  # Separate sparse matrix
    else:
        combined_features_sparse = combined_features
    if not issparse(target_vector):
        target_vector = csr_matrix(target_vector)

    similarities = cosine_similarity(target_vector, combined_features_sparse)
    top_indices = similarities[0].argsort()[::-1][1:k + 1]

    recommendations = []
    for i in top_indices:
        book_index = original_index[i]  # Use original index
        book_info = book_data.iloc[book_index]  # Use .iloc with index
        recommendations.append({
            'book_title': book_info['book_title'],
            'url': book_info[url_column]
        })

    return recommendations

def combine_recommendations(ratings, category, aged, content, weight_ratings, weight_category, weight_aged, weight_content, book_data, url_column='img_l'):
    """Combines recommendations with 5 books from each source and retrieves URLs efficiently."""

    def safe_extract_titles(data):
        if not isinstance(data, list):
            print(f"Warning: Expected a list, but got {type(data)} -> {data}")
            return []
        return [item['book_title'] for item in data if isinstance(item, dict) and 'book_title' in item]

    # Extract titles from each recommendation list
    ratings_list = safe_extract_titles(ratings)
    category_list = safe_extract_titles(category)
    aged_list = safe_extract_titles(aged)
    content_list = safe_extract_titles(content)

    # Define how many recommendations to take from each source (5 from each)
    num_books_per_source = 5

    # Create the combined recommendations by taking 5 books from each list
    combined_recommendations = []

    # Add books from each recommendation source with their corresponding weights
    for rec_list, weight in zip([ratings_list, category_list, aged_list, content_list], 
                                [weight_ratings, weight_category, weight_aged, weight_content]):
        # Take the top 5 books from each list
        weighted_recommendations = rec_list[:num_books_per_source]
        for rec in weighted_recommendations:
            combined_recommendations.append((rec, weight))

    # Remove duplicates by converting combined recommendations to a dictionary
    # This will ensure no book repeats, and the last one will be kept
    combined_recommendations_dict = dict(combined_recommendations)

    # Log the combined recommendations before applying URLs
    logging.debug(f"Combined recommendations (without sorting): {combined_recommendations_dict}")

    # Get URLs for the recommended books from the book data
    recommended_books = book_data[book_data['book_title'].isin(combined_recommendations_dict.keys())]
    book_url_map = dict(zip(recommended_books['book_title'], recommended_books[url_column]))

    # Prepare the final list of recommendations with URLs
    final_recommendations = [{'book_title': rec, 'url': book_url_map.get(rec)} 
                             for rec in combined_recommendations_dict.keys()]

    logging.debug(f"Final recommendations with URLs: {final_recommendations}")

    return final_recommendations



def get_recommendations_for_book(book_title, combined_features, book_data, search_type, k=10):
    """Gets recommendations for a book based on search type, with logging."""

    if search_type == 'hybrid':
        content_recs = get_book_recommendations_content(book_title, combined_features, book_data, k)
        logging.debug(f"Hybrid - Content recommendations: {content_recs}")
        
        ratings_recs = get_recommendations_ratings(book_title, item_user_matrix, book_data, k)
        logging.debug(f"Hybrid - Ratings recommendations: {ratings_recs}")

        category_recs = get_recommendations_category(book_title, category_matrix, book_data, k)
        logging.debug(f"Hybrid - Category recommendations: {category_recs}")

        aged_recs = get_recommendations_aged(book_title, aged_matrix, book_data, k)
        logging.debug(f"Hybrid - Aged recommendations: {aged_recs}")

        hybrid_recs = combine_recommendations(ratings_recs, category_recs, aged_recs, content_recs, 1, 1, 1, 1, book_data)
        logging.debug(f"Hybrid - Combined recommendations: {hybrid_recs}")
        return hybrid_recs

    elif search_type == 'ratings':
        ratings_recs = get_recommendations_ratings(book_title, item_user_matrix, book_data, k)
        logging.debug(f"Ratings recommendations: {ratings_recs}")
        return ratings_recs

    elif search_type == 'category':
        category_recs = get_recommendations_category(book_title, category_matrix, book_data, k)
        logging.debug(f"Category recommendations: {category_recs}")
        # Convert category_matrix to list of dictionaries before returning
        return category_recs #This is the change.

    elif search_type == 'aged':
        aged_recs = get_recommendations_aged(book_title, aged_matrix, book_data, k)
        logging.debug(f"Aged recommendations: {aged_recs}")
        return aged_recs

    elif search_type == 'content':
        content_recs = get_book_recommendations_content(book_title, combined_features, book_data, k)
        logging.debug(f"Content recommendations: {content_recs}")
        return content_recs

    else:
        return []

def get_recommendations(user_input, search_type, item_user_matrix, category_matrix, aged_matrix, combined_features, book_data, k=20):
    """Gets recommendations based on user input and search type, with improved handling."""

    best_matches = find_best_matches(user_input, book_data)  # Assuming find_best_matches is defined elsewhere

    if not best_matches:
        return [{"book_title": "No matching book found.", "url": None}]

    all_recommendations = OrderedDict()  # Use OrderedDict to maintain order and prevent duplicates

    for best_match in best_matches:
        recs = get_recommendations_for_book(best_match, combined_features, book_data, search_type, k)

        if isinstance(recs, list):  # Ensure recs is a list before iterating
            for rec in recs:
                if isinstance(rec, dict) and "book_title" in rec:
                    title = rec["book_title"]
                    if title not in all_recommendations:
                        all_recommendations[title] = rec  # Add only if not already present
                else:
                    print(f"Warning: Unexpected recommendation format: {rec}")
        else:
            print(f"Warning: get_recommendations_for_book returned non-list: {recs}")

    return list(all_recommendations.values())  # Return a list of unique recommendations


def find_best_matches(query, book_data):
    """Finds the best matching book titles using fuzzy matching."""
    if not isinstance(query, str) or not query:  # Handle empty or non-string queries
        return []

    titles = book_data['book_title'].tolist()
    matches = process.extract(query, titles, limit=10)

    best_matches = [match[0] for match in matches if match[1] > 60]
    return best_matches


def get_search_results(query, book_data_admin, book_data):  # Add book_data as argument
    """Gets the book data for the best matching titles."""

    best_matches = find_best_matches(query, book_data)  # Pass book_data here

    if best_matches:
        results = book_data_admin[book_data_admin['book_title'].isin(best_matches)].to_dict(orient='records')
        return results
    return []


def train_model(df):
    """Train a model for cosine similarity using TF-IDF vectorization."""
    tfidf_category = TfidfVectorizer(stop_words="english")
    category_matrix_new = tfidf_category.fit_transform(df["Category"].fillna(""))

    tfidf_summary = TfidfVectorizer(stop_words="english", max_features=5000)
    summary_matrix = tfidf_summary.fit_transform(df["Summary"].fillna(""))

    combined_matrix = hstack([category_matrix_new, summary_matrix])

    return (tfidf_category, tfidf_summary, combined_matrix, df)

def recommend_books(categories, tfidf_category, tfidf_summary, combined_matrix, df, top_n=8):
    """Recommend books based on cosine similarity."""
    try:
        if not categories or not isinstance(categories, list):
            return []

        input_cat = " ".join(categories)
        input_category_vector = tfidf_category.transform([input_cat])
        input_summary_vector = tfidf_summary.transform([""])


        input_vector = hstack([input_category_vector, input_summary_vector])

        similarities = cosine_similarity(input_vector, combined_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]

        recommendations = df.iloc[top_indices].copy()
        recommendations["similarity"] = similarities[top_indices]

        columns_to_keep = ["book_title", "Category", "rating", "img_m", "similarity"]
        recommendations = recommendations[columns_to_keep]

        return recommendations.to_dict(orient="records")

    except Exception as e:
        print(f"Recommendation error: {e}")
        return []