"""
Student Implementation Adapter for Music Recommender Web API
=============================================================
This file bridges your notebook implementation with the Flask web application.

INSTRUCTIONS:
1. Copy your COMPLETE, TESTED implementations below
2. Do NOT modify the get_recommendations_for_api function
3. Test using: python -m utils.test_student_adapter
"""

import numpy as np
import pandas as pd
import os

# ============================================================================
# STUDENT IMPLEMENTATION SECTION
# Copy your complete, final implementations from the notebook below
# ============================================================================

class FeatureScaler:
    """
    Standard scaler for normalizing features to mean=0, std=1
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self._fitted = False
    
    def fit(self, X):
        """Learn the scaling parameters from data X."""
        # calculate mean and std for each feature
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # replace any zero std with 1 to avoid division by zero
        self.std[self.std == 0] = 1
        self._fitted = True
    
    def transform(self, X):
        """Apply the learned scaling to data X."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        # standardize the features
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class KNNRecommender:
    """
    K-Nearest Neighbors recommender for music
    """
    
    def __init__(self, k=10):
        self.k = k
        self.item_profile = None
        self.features_matrix = None
        self.feature_columns = None
        self.track_id_to_index = {}
        self._fitted = False
    
    @staticmethod
    def euclidean_distance(a, b):
        """Calculate Euclidean distance between vectors a and b."""
        # compute euclidean distance
        return np.sqrt(np.sum((a - b) ** 2))
    
    @staticmethod
    def cosine_distance(a, b):
        """Calculate Cosine distance between vectors a and b."""
        # compute cosine distance: 1 - cosine_similarity
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # handle zero vectors
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        # cosine similarity = dot product / (norm_a * norm_b)
        cosine_sim = np.dot(a, b) / (norm_a * norm_b)
        # cosine distance = 1 - cosine similarity
        return 1 - cosine_sim
    
    def fit(self, item_profile_df, feature_columns):
        """Prepare the recommender with track data."""
        # drop rows with NaN values in feature columns to prevent NaN distances
        item_profile_df = item_profile_df.dropna(subset=feature_columns)
        self.item_profile = item_profile_df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.features_matrix = self.item_profile[self.feature_columns].values
        self.track_id_to_index = {track_id: i for i, track_id in enumerate(self.item_profile['id'])}
        self._fitted = True
        print(f"Fit complete. Loaded {len(self.item_profile)} tracks.")
    
    def find_neighbors(self, track_id, n_neighbors=None, distance_metric='euclidean'):
        """Find k nearest neighbors for a track."""
        if n_neighbors is None:
            n_neighbors = self.k
        
        # select distance function
        distance_functions = {
            'euclidean': self.euclidean_distance,
            'cosine': self.cosine_distance
        }
        
        if distance_metric not in distance_functions:
            raise ValueError(f"Unknown metric: {distance_metric}")
        
        if track_id not in self.track_id_to_index:
            raise ValueError(f"Track ID {track_id} not found.")
        
        distance_func = distance_functions[distance_metric]
        
        # get query track index and features
        query_index = self.track_id_to_index[track_id]
        query_features = self.features_matrix[query_index]
        
        # calculate distances to all other tracks
        distances = []
        for i, track_features in enumerate(self.features_matrix):
            # skip the query track itself
            if i == query_index:
                continue
            
            # calculate distance
            dist = distance_func(query_features, track_features)
            track_id_at_i = self.item_profile.iloc[i]['id']
            distances.append((dist, track_id_at_i))
        
        # sort by distance and return top n
        distances.sort(key=lambda x: x[0])
        return distances[:n_neighbors]
    
    def recommend(self, track_id, n_recommendations=None, distance_metric='euclidean'):
        """Generate recommendations for a track."""
        if self.item_profile is None:
            raise RuntimeError("Recommender has not been fitted.")
        
        neighbors = self.find_neighbors(track_id, n_recommendations, distance_metric)
        
        # get dataframe for neighbor tracks
        neighbor_ids = [tid for distance, tid in neighbors]
        results_df = self.item_profile[self.item_profile['id'].isin(neighbor_ids)].copy()
        
        # add distance column
        distances_map = {tid: dist for dist, tid in neighbors}
        results_df['distance'] = results_df['id'].map(distances_map)
        
        return results_df.sort_values('distance')
    
    def recommend_from_vector(self, query_vector, n_recommendations=None, 
                            distance_metric='cosine', selected_features=None):
        """
        Find songs similar to a custom feature vector (profile-based recommendation)
        """
        if self.item_profile is None:
            raise RuntimeError("Recommender has not been fitted.")
        
        if n_recommendations is None:
            n_recommendations = self.k
        
        # select distance function
        distance_functions = {
            'euclidean': self.euclidean_distance,
            'cosine': self.cosine_distance
        }
        
        if distance_metric not in distance_functions:
            raise ValueError(f"Unknown metric: {distance_metric}")
        
        distance_func = distance_functions[distance_metric]
        
        # handle feature selection
        if selected_features is not None:
            # get indices of selected features
            feature_indices = [i for i, f in enumerate(self.feature_columns) 
                             if f in selected_features]
            if not feature_indices:
                raise ValueError("No valid features selected")
            
            # slice matrices to selected features
            features_subset = self.features_matrix[:, feature_indices]
            query_subset = query_vector[feature_indices] if len(query_vector) > len(feature_indices) else query_vector
        else:
            # use all features
            features_subset = self.features_matrix
            query_subset = query_vector
        
        # calculate distances to all tracks
        distances = []
        for i in range(len(self.item_profile)):
            track_features = features_subset[i]
            dist = distance_func(query_subset, track_features)
            track_id = self.item_profile.iloc[i]['id']
            distances.append((dist, track_id))
        
        # sort by distance and get top n
        distances.sort(key=lambda x: x[0])
        top_n = distances[:n_recommendations]
        
        # create result dataframe
        neighbor_ids = [tid for dist, tid in top_n]
        results_df = self.item_profile[self.item_profile['id'].isin(neighbor_ids)].copy()
        
        # add distance column
        distances_map = {tid: dist for dist, tid in top_n}
        results_df['distance'] = results_df['id'].map(distances_map)
        
        return results_df.sort_values('distance')


# custom hybrid distance function
def custom_hybrid_distance(track_a_data, track_b_data, audio_features_a, audio_features_b, w_artist=0.5):
    """
    Hybrid distance combining audio features and artist metadata
    """
    # calculate audio distance using cosine distance
    norm_a = np.linalg.norm(audio_features_a)
    norm_b = np.linalg.norm(audio_features_b)
    
    if norm_a == 0 or norm_b == 0:
        audio_distance = 1.0
    else:
        cosine_sim = np.dot(audio_features_a, audio_features_b) / (norm_a * norm_b)
        audio_distance = 1 - cosine_sim
    
    # artist similarity bonus: 0 if same artist, 1 if different
    artist_a = track_a_data.get('artist', '')
    artist_b = track_b_data.get('artist', '')
    artist_distance = 0.0 if artist_a == artist_b else 1.0
    
    # combine distances with weights
    # higher w_artist favors same artist, lower w_artist favors variety
    hybrid_distance = (1 - w_artist) * audio_distance + w_artist * artist_distance
    
    return hybrid_distance


class HybridKNNRecommender(KNNRecommender):
    """
    Hybrid KNN recommender combining audio features and metadata
    """
    
    def find_neighbors(self, track_id, n_neighbors=None, distance_metric='hybrid', w_artist=0.5):
        """
        Find neighbors using hybrid distance that combines audio features and metadata
        """
        if distance_metric != 'hybrid':
            return super().find_neighbors(track_id, n_neighbors, distance_metric)
        
        if n_neighbors is None:
            n_neighbors = self.k
        
        if track_id not in self.track_id_to_index:
            raise ValueError(f"Track ID {track_id} not found.")
        
        # get query track data
        query_index = self.track_id_to_index[track_id]
        query_features = self.features_matrix[query_index]
        query_data = self.item_profile.iloc[query_index].to_dict()
        
        # calculate hybrid distances to all other tracks
        distances = []
        for i in range(len(self.item_profile)):
            # skip the query track itself
            if i == query_index:
                continue
            
            # get candidate track data
            candidate_features = self.features_matrix[i]
            candidate_data = self.item_profile.iloc[i].to_dict()
            
            # calculate hybrid distance
            dist = custom_hybrid_distance(
                query_data, candidate_data,
                query_features, candidate_features,
                w_artist=w_artist
            )
            
            track_id_at_i = self.item_profile.iloc[i]['id']
            distances.append((dist, track_id_at_i))
        
        # sort by distance and return top n
        distances.sort(key=lambda x: x[0])
        return distances[:n_neighbors]



# ============================================================================
# API ADAPTER SECTION - DO NOT MODIFY ANYTHING BELOW THIS LINE
# ============================================================================

# Cache for the recommender instance
_recommender_cache = None
_audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                   'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']


def get_recommendations_for_api(track_id, k=10, metric='cosine', use_hybrid=False):
    """
    Bridge function between student implementation and web API.
    DO NOT MODIFY THIS FUNCTION.
    """
    global _recommender_cache
    
    try:
        # Load data and initialize recommender if needed
        if _recommender_cache is None:
            # Find the data file
            possible_paths = [
                'data/mergedFile.csv',
                '../data/mergedFile.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'mergedFile.csv'),
                'data/item_profile.csv',
                '../data/item_profile.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'item_profile.csv')
            ]
            
            item_profile = None
            for path in possible_paths:
                if os.path.exists(path):
                    item_profile = pd.read_csv(path, dtype={'id': str})
                    break
            
            if item_profile is None:
                raise FileNotFoundError("Could not find mergedFile.csv or item_profile.csv")
            
            # Initialize the appropriate recommender
            if use_hybrid and 'HybridKNNRecommender' in globals():
                _recommender_cache = HybridKNNRecommender(k=k)
            else:
                _recommender_cache = KNNRecommender(k=k)
            
            # Use only features that exist in the loaded file
            available_feats = [f for f in _audio_features if f in item_profile.columns]
            if not available_feats:
                raise ValueError("No required audio feature columns found in data file")

            _recommender_cache.fit(item_profile, available_feats)
            print(f"Initialized {type(_recommender_cache).__name__} with {len(item_profile)} tracks")
        
        # Get recommendations
        recommendations = _recommender_cache.recommend(
            track_id, 
            n_recommendations=k, 
            distance_metric=metric
        )
        
        # Convert to API format
        result = {}
        for _, row in recommendations.iterrows():
            result[row['id']] = {
                'distance': float(row['distance']),
                'song': row.get('song', 'Unknown'),
                'artist': row.get('artist', 'Unknown'),
                'features': {feat: float(row.get(feat, 0)) for feat in _audio_features if feat in row}
            }
        
        return result
        
    except Exception as e:
        print(f"Error in student implementation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_implementation():
    """
    Test function to verify your implementation works.
    Run this after copying your code above.
    """
    try:
        from utils.test_student_adapter import run_comprehensive_tests
        return run_comprehensive_tests()
    except ImportError:
        # Fallback if test file is not in expected location
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-m', 'utils.test_student_adapter'], 
                              capture_output=False)
        return result.returncode == 0


if __name__ == "__main__":
    print("To test your implementation, run:")
    print("  python -m utils.test_student_adapter")             

# ============================================================================
# AUTO-INITIALIZATION FOR API INTEGRATION
# ============================================================================

def initialize_for_api():
    """
    Initialize the recommender for API usage.
    This is called when api_helpers imports this module.
    """
    import os
    import pandas as pd
    
    # Try to find and load the data (prefer mergedFile.csv which has full feature set)
    possible_paths = [
        'data/mergedFile.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'mergedFile.csv'),
        'data/item_profile.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'item_profile.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype={'id': str})
                audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                                 'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']
                
                # Create and fit the recommender
                recommender = KNNRecommender(k=10)
                recommender.fit(df, audio_features)
                
                print(f"✅ Student recommender initialized with {len(df)} tracks")
                return recommender
                
            except Exception as e:
                print(f"Error loading data from {path}: {e}")
                continue
    
    print("⚠️ Could not initialize student recommender - data files not found")
    return None

# Export the initialized recommender for api_helpers to use (disabled to avoid duplicate init; api_helpers handles it)
# student_recommender_instance = initialize_for_api()