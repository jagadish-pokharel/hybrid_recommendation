import pandas as pd
import numpy as np
#from .utily import load_models

from .utily import item_user_matrix,knn_ratings,catogery_matrix,knn_category,aged_matrix,combined_features,book_data


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


def get_recommendations_ratings(item, item_user_matrix, k):
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        return dot_product / (norm_vec1 * norm_vec2)

    def find_nearest_neighbors(target_vector, item_user_matrix, k):
        similarities = []

        for idx, row in item_user_matrix.iterrows():
            # Calculate cosine similarity
            similarity = cosine_similarity(target_vector, row.values)
            similarities.append((idx, similarity))
        
        # Sort by similarity (in descending order) and select the top k + 1 nearest neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = similarities[:k + 1]

        # Extract indices and similarities
        indices = [item_user_matrix.index.get_loc(x[0]) for x in nearest_neighbors if x[0] != target_item]
        similarities = [x[1] for x in nearest_neighbors if x[0] != target_item]
        
        return similarities, indices

    # Check if item is in the DataFrame index
    if item not in item_user_matrix.index:
        return [f"Item '{item}' is not in the dataset."]

    # Extract the feature vector for the item
    target_vector = item_user_matrix.loc[item].values
    global target_item
    target_item = item

    # Compute nearest neighbors using cosine similarity
    similarities, indices = find_nearest_neighbors(target_vector, item_user_matrix, k)

    # Collect recommendations (excluding the input item itself)
    recommendations = []
    for i in range(len(indices)):
        recommendations.append(item_user_matrix.index[indices[i]])

    return recommendations




###  CATEGORY ###




# def find_nearest_neighbors(target_vector, item_user_matrix, k):
#     similarities = []
    
#     for idx, row in catogery_matrix.iterrows():
#         # Calculate cosine similarity
#         similarity = cosine_similarity(target_vector, row.values)
#         similarities.append((idx, similarity))
    
#     # Sort by similarity (in descending order) and select the top k + 1 nearest neighbors
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     nearest_neighbors = similarities[:k + 1]
    
#     # Extract indices and similarities
#     indices = [catogery_matrix.index.get_loc(x[0]) for x in nearest_neighbors if x[0] != target_item]
#     similarities = [x[1] for x in nearest_neighbors if x[0] != target_item]
    
#     return similarities, indices


# def get_recommendations_category(item, k):
#     # Check if item is in the DataFrame index
#     if item not in catogery_matrix.index:
#         return [f"Item '{item}' is not in the dataset."]
    
#     # Extract the feature vector for the item
#     target_vector = catogery_matrix.loc[item].values
#     global target_item
#     target_item = item

#     # Compute nearest neighbors using cosine similarity
#     similarities, indices = find_nearest_neighbors(target_vector, catogery_matrix, k)
    
#     # Collect recommendations (excluding the input item itself)
#     recommendations = []
#     for i in range(len(indices)):
#         recommendations.append(catogery_matrix.index[indices[i]])
    
#     return recommendations






def get_recommendations_category(item, catogery_matrix, k):
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        return dot_product / (norm_vec1 * norm_vec2)

    def find_nearest_neighbors(target_vector, catogery_matrix, k):
        similarities = []

        for idx, row in catogery_matrix.iterrows():
            # Calculate cosine similarity
            similarity = cosine_similarity(target_vector, row.values)
            similarities.append((idx, similarity))

        # Sort by similarity (in descending order) and select the top k + 1 nearest neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = similarities[:k + 1]

        # Extract indices and similarities
        indices = [catogery_matrix.index.get_loc(x[0]) for x in nearest_neighbors if x[0] != target_item]
        similarities = [x[1] for x in nearest_neighbors if x[0] != target_item]

        return similarities, indices

    # Check if item is in the DataFrame index
    if item not in catogery_matrix.index:
        return [f"Item '{item}' is not in the dataset."]

    # Extract the feature vector for the item
    target_vector = catogery_matrix.loc[item].values
    global target_item
    target_item = item

    # Compute nearest neighbors using cosine similarity
    similarities, indices = find_nearest_neighbors(target_vector, catogery_matrix, k)

    # Collect recommendations (excluding the input item itself)
    recommendations = []
    for i in range(len(indices)):
        recommendations.append(catogery_matrix.index[indices[i]])

    return recommendations


###   AGED ####


def get_recommendations_aged(item, aged_matrix, k):
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        return dot_product / (norm_vec1 * norm_vec2)

    def find_nearest_neighbors(target_vector, aged_matrix, k):
        similarities = []

        for idx, row in aged_matrix.iterrows():
            # Calculate cosine similarity
            similarity = cosine_similarity(target_vector, row.values)
            similarities.append((idx, similarity))

        # Sort by similarity (in descending order) and select the top k + 1 nearest neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = similarities[:k + 1]

        # Extract indices and similarities
        indices = [aged_matrix.index.get_loc(x[0]) for x in nearest_neighbors if x[0] != target_item]
        similarities = [x[1] for x in nearest_neighbors if x[0] != target_item]

        return similarities, indices

    # Check if item is in the DataFrame index
    if item not in aged_matrix.index:
        return [f"Item '{item}' is not in the dataset."]

    # Extract the feature vector for the item
    target_vector = aged_matrix.loc[item].values
    global target_item
    target_item = item

    # Compute nearest neighbors using cosine similarity
    similarities, indices = find_nearest_neighbors(target_vector, aged_matrix, k)

    # Collect recommendations (excluding the input item itself)
    recommendations = []
    for i in range(len(indices)):
        recommendations.append(aged_matrix.index[indices[i]])

    return recommendations




#### import BOOK title combined_features and book_data pickle jasari aagi ko jasto

def get_book_recommendations_content(book_title, combined_features, book_data,k):
    def custom_cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        return dot_product / (norm_vec1 * norm_vec2)

    def find_nearest_neighbors(target_vector, combined_features, k):
        similarities = []

        for idx in range(combined_features.shape[0]):  # Use shape[0] to get the number of rows
            # Calculate cosine similarity
            similarity = custom_cosine_similarity(target_vector, combined_features[idx].toarray().flatten())
            similarities.append((idx, similarity))
        
        # Sort by similarity (in descending order) and select the top k + 1 nearest neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = similarities[:k + 1]

        # Extract indices and similarities
        indices = [idx for idx, sim in nearest_neighbors if idx != target_index]
        similarities = [sim for idx, sim in nearest_neighbors if idx != target_index]
        
        return similarities, indices

    # Check if the book title exists in the DataFrame
    if book_title not in book_data['book_title'].values:
        return f"Book title '{book_title}' is not in the dataset."

    # Find the index of the book based on the title
    global target_index
    target_index = book_data[book_data['book_title'] == book_title].index[0]
    
    # Convert the sparse matrix to dense format
    target_vector = combined_features[target_index].toarray().flatten()
    
    # Compute nearest neighbors using cosine similarity
    similarities, indices = find_nearest_neighbors(target_vector, combined_features, k)

    # Collect recommendations (excluding the input book itself)
    recommendations = []
    for i in range(len(indices)):
        recommendations.append(book_data['book_title'].iloc[indices[i]])

    return recommendations






# def combine_recommendations(ratings, category, aged, content, weight_ratings, weight_category, weight_aged, weight_content, book_data):
#     all_recommendations = set(ratings) | set(category) | set(aged) | set(content)
#     recommendation_scores = {}

#     for rec in all_recommendations:
#         score = 0
#         recurring_count = 0

#         if rec in ratings:
#             score += weight_ratings
#             recurring_count += 1
#         if rec in category:
#             score += weight_category
#             recurring_count += 1
#         if rec in aged:
#             score += weight_aged
#             recurring_count += 1
#         if rec in content:
#             score += weight_content
#             recurring_count += 1

#         # Adjust the score by the number of times the item recurs
#         recommendation_scores[rec] = score * recurring_count

#     # Sort recommendations by score in descending order and get the top 5
#     sorted_recommendations = sorted(recommendation_scores, key=recommendation_scores.get, reverse=True)[:5]

#     # Access the cover URL from the book_data
#     combined_recommendations = [
#         rec
#        #"img_l": book_data.get(rec, {}).get("img_l", "No cover available")  # Using large-sized cover URL, or a placeholder if not found
    
#         for rec in sorted_recommendations
#     ]

#     return combined_recommendations



def combine_recommendations(ratings, category, aged, content, weight_ratings, weight_category, weight_aged, weight_content, book_data):
    # Initialize a dictionary to hold recommendations and scores
    recommendation_scores = {}
    all_recommendations = set(ratings) | set(category) | set(aged) | set(content)
    
    for rec in all_recommendations:
        score = 0
        recurring_count = 0
        
        # If the book is in the ratings system, add its weight to the score
        if rec in ratings:
            score += weight_ratings
            recurring_count += 1
        # If the book is in the category system, add its weight to the score
        if rec in category:
            score += weight_category
            recurring_count += 1
        # If the book is in the aged system, add its weight to the score
        if rec in aged:
            score += weight_aged
            recurring_count += 1
        # If the book is in the content-based recommendation system, add its weight to the score
        if rec in content:
            score += weight_content
            recurring_count += 1

        # Adjust the score by the number of times the item recurs in different systems
        if recurring_count > 0:
            recommendation_scores[rec] = score * recurring_count

    # Sort recommendations by score in descending order and get the top 5
    sorted_recommendations = sorted(recommendation_scores, key=recommendation_scores.get, reverse=True)[:5]
    
   
    combined_recommendations = [
        rec
        for rec in sorted_recommendations
    ]
    
    return combined_recommendations
