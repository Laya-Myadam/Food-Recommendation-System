
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'nutrients_csvfile.csv'  # Replace with actual path in Colab
df = pd.read_csv(file_path)

# Explore the dataset
print(df.head())
print(df.describe())
print(df.info())

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Remove missing values
df = df.dropna()

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\W', ' ', text)  # Remove non-word characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    return text  # Return the text as is if not a string

# Apply preprocessing to all text-based columns
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].apply(preprocess_text)

# Check the result
print(df.head())

"""#Visualizations

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


numeric_columns = ['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# 1. Bar Chart for Nutritional Components
df.set_index('Food').drop('Category', axis=1).plot(kind='bar', figsize=(10, 6))
plt.title('Nutritional Components for Different Food Items')
plt.ylabel('Value')
plt.xlabel('Food Item')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Pie Chart for Category Distribution
category_counts = df['Category'].value_counts()
category_counts.plot(kind='pie', figsize=(8, 8), autopct='%1.1f%%', startangle=90)
plt.title('Food Category Distribution')
plt.ylabel('')
plt.show()

# 3. Correlation Heatmap (only numeric columns)
corr = df[numeric_columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Nutritional Components')
plt.show()

# 4. Box Plot for Nutritional Values
df.drop(['Food', 'Category'], axis=1).plot(kind='box', figsize=(10, 6))
plt.title('Box Plot of Nutritional Components')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Scatter Plot for Carbs vs. Calories
sns.scatterplot(x='Carbs', y='Calories', data=df, hue='Category', s=100)
plt.title('Scatter Plot: Carbs vs Calories')
plt.xlabel('Carbs (g)')
plt.ylabel('Calories')
plt.show()

from textblob import TextBlob

# Function to apply TextBlob spelling correction
def correct_text(text):
    try:
        return str(TextBlob(text).correct())
    except Exception as e:
        return text  # Return the original text if any error occurs

# Iterate over all columns and apply TextBlob to those that contain strings
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is of type 'object' (i.e., text)
        df[column] = df[column].apply(correct_text)

# Print the DataFrame to check the result
print(df.head())

df.head()

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization on 'Food' column or any other relevant text column
if 'Food' in df.columns: 
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(df['Food'])  # Applying TF-IDF on the 'Food' column
    feature_names = vectorizer.get_feature_names_out()

    # Print top keywords for the first record
    first_record = tfidf_matrix[0]  # First record in the TF-IDF matrix
    top_indices = first_record.toarray().argsort()[0, -10:]  # Get indices of the top 10 words
    top_keywords = [feature_names[i] for i in top_indices] 
    
    print('Top keywords for the first record: ' + str(top_keywords))

def process_food_data(df):
    print("Starting processing...")
    print("Input DataFrame shape:", df.shape)

    # Separate text and numeric columns
    text_col = 'Food'
    numeric_cols = ['Grams', 'Calories', 'Protein', 'Fat', 'Sat.Fat',
                   'Fiber', 'Carbs']

    print("\nProcessing Food column with TF-IDF...")
    # TF-IDF for Food column
    tfidf = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'\b\w+\b',
        stop_words='english',
        min_df=2,
        max_features=100
    )

    # Transform Food column
    food_tfidf = tfidf.fit_transform(df[text_col])
    print("TF-IDF shape:", food_tfidf.shape)

    print("\nProcessing numeric columns...")
    # Clean and convert numeric columns
    df_numeric = df[numeric_cols].copy()
    for col in numeric_cols:
        # Remove spaces and convert to float
        df_numeric[col] = df_numeric[col].astype(str).str.replace(' ', '').astype(float)

    # Scale numeric columns
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(df_numeric)
    print("Numeric features shape:", numeric_features.shape)

    # Combine TF-IDF and numeric features
    combined_features = np.hstack([
        food_tfidf.toarray(),
        numeric_features
    ])

    # Create feature names
    feature_names = (
        list(tfidf.get_feature_names_out()) +
        numeric_cols
    )

    print("\nFinal outputs:")
    print("Shape of final features array:", combined_features.shape)
    print("Number of feature names:", len(feature_names))
    print("First few feature names:", feature_names[:10])
    print("Sample of processed data (first row):", combined_features[0][:10])

    return combined_features, feature_names

# Call the function
features, feature_names = process_food_data(df)

def search_and_filter_foods(df,
                          search_term=None,
                          category=None,
                          min_protein=None,
                          max_calories=None,
                          min_fiber=None,
                          max_fat=None):
    # Create a copy of the DataFrame
    result = df.copy()

    # Convert numeric columns to float, handling any spaces in strings
    numeric_cols = ['Calories', 'Protein', 'Fat', 'Fiber']
    for col in numeric_cols:
        result[col] = result[col].astype(str).str.replace(' ', '').astype(float)

    # Apply text search if specified
    if search_term:
        result = result[result['Food'].str.contains(search_term, case=False)]

    # Apply category filter if specified
    if category:
        result = result[result['Category'] == category]

    # Apply numeric filters if specified
    if min_protein is not None:
        result = result[result['Protein'] >= min_protein]

    if max_calories is not None:
        result = result[result['Calories'] <= max_calories]

    if min_fiber is not None:
        result = result[result['Fiber'] >= min_fiber]

    if max_fat is not None:
        result = result[result['Fat'] <= max_fat]

    return result

def get_unique_categories(df):
    """Get list of unique categories in the dataset."""
    return sorted(df['Category'].unique())

def get_nutrient_ranges(df):
    # Convert numeric columns to float
    numeric_cols = ['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']
    df_numeric = df.copy()

    for col in numeric_cols:
        df_numeric[col] = df_numeric[col].astype(str).str.replace(' ', '').astype(float)

    ranges = {}
    for col in numeric_cols:
        ranges[col] = {
            'min': df_numeric[col].min(),
            'max': df_numeric[col].max(),
            'mean': df_numeric[col].mean()
        }

    return ranges

# Example usage:
def example_queries(df):
    """Show example queries using the search and filter functions."""
    print("Example queries:")

    # 1. High protein foods (>20g)
    high_protein = search_and_filter_foods(df, min_protein=20)
    print("\nHigh protein foods (>20g):")
    print(high_protein[['Food', 'Protein', 'Calories']].head())

    # 2. Low calorie foods in Vegetables category
    low_cal_veggies = search_and_filter_foods(df,
                                            category='Vegetables',
                                            max_calories=50)
    print("\nLow calorie vegetables (<50 cal):")
    print(low_cal_veggies[['Food', 'Calories', 'Category']].head())

    # 3. Search for "chicken" with specific nutrient ranges
    chicken_search = search_and_filter_foods(df,
                                           search_term='chicken',
                                           max_fat=10,
                                           min_protein=15)
    print("\nChicken dishes (fat<10g, protein>15g):")
    print(chicken_search[['Food', 'Protein', 'Fat']].head())

# Example of how to use the functions:
if __name__ == "__main__":
    # Assuming df is your DataFrame
    # Get available categories
    categories = get_unique_categories(df)
    print("Available categories:", categories)

    # Get nutrient ranges
    ranges = get_nutrient_ranges(df)
    print("\nNutrient ranges:", ranges)

    # Run example queries
    example_queries(df)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

def clean_features(features):
    """
    Clean features by handling NaN values
    """
    # Initialize imputer
    imputer = SimpleImputer(strategy='mean')

    # Fit and transform the features
    cleaned_features = imputer.fit_transform(features)

    return cleaned_features
def precision_at_k(y_true, y_pred, k):
    """
    Calculate Precision@K
    y_true: list of true relevant items
    y_pred: list of predicted items (top-K)
    k: number of top recommendations
    """
    top_k_pred = y_pred[:k]
    relevant_recommendations = sum([1 for true, pred in zip(y_true, top_k_pred) if true == pred])
    precision = relevant_recommendations / k
    return precision

def recall_at_k(y_true, y_pred, k):
    """
    Calculate Recall@K
    y_true: list of true relevant items
    y_pred: list of predicted items (top-K)
    k: number of top recommendations
    """
    top_k_pred = y_pred[:k]
    relevant_recommendations = sum([1 for true in y_true if true in top_k_pred])
    recall = relevant_recommendations / len(y_true)  # Recall is the fraction of true relevant items found in top-K
    return recall

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE)
    y_true: list of true ratings (e.g., user ratings)
    y_pred: list of predicted ratings
    """
    differences = np.square(np.subtract(y_true, y_pred))
    rmse_value = np.sqrt(np.mean(differences))
    return rmse_value

def evaluate_feature_representation(features, feature_names, df, n_clusters=5):
    """
    Evaluate the quality of the feature representation using multiple metrics.
    """
    print("Feature Representation Evaluation")
    print("-" * 40)

    # Clean features first
    features = clean_features(features)

    # 1. All the Basic Statistics
    print("\n1. Feature Statistics:")
    print("Number of samples: {}".format(features.shape[0]))
    print("Number of features: {}".format(features.shape[1]))


    # 2. Sparsity-Analysis
    sparsity = 1.0 - np.count_nonzero(features) / features.size
    print("\n2. Feature Sparsity: {:.2%}".format(sparsity))


    # 3. Clustering-Evaluation
    print("\n3. Clustering Metrics:")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)

    # Silhouette-Score
    sil_score = silhouette_score(features, clusters)
    print("Silhouette Score: {:.3f}".format(sil_score))


    # Calinski-Harabasz-Index
    ch_score = calinski_harabasz_score(features, clusters)
    print("Calinski-Harabasz Score: {:.3f}".format(ch_score))

    #  Classification Metrics (if 'Category' is available)
    if 'Category' in df.columns:
        # Align clusters with true categories
        cluster_category_map = {}
        for i in range(n_clusters):
            # Get the most frequent category for each cluster
            cluster_data = df[clusters == i]
            most_common_category = cluster_data['Category'].mode()[0]
            cluster_category_map[i] = most_common_category

        # Map clusters to categories
        predicted_labels = np.array([cluster_category_map[cluster] for cluster in clusters])
        true_labels = df['Category'].values

        # Calculate precision, recall, F1 score, and accuracy
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_labels, predicted_labels)

        print("\n4. Classification Metrics:")
        print("Precision (Weighted): {:.3f}".format(precision))
        print("Recall (Weighted): {:.3f}".format(recall))
        print("F1 Score (Weighted): {:.3f}".format(f1))
        print("Accuracy: {:.3f}".format(accuracy))

        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        print("\nConfusion Matrix:")
        print(cm)

    # 4. Feature Importance Analysis
    print("\n4. Top Important Features per Cluster:")
    cluster_centers = kmeans.cluster_centers_
    for i in range(n_clusters):
        center = cluster_centers[i]
        top_indices = np.argsort(center)[-5:]
        top_features = [(feature_names[j], center[j]) for j in top_indices]
        print("\nCluster {} top features:".format(i))
        for feat, val in top_features:
            print("{}: {:.3f}".format(feat, val))

    # 5. Dimensionality Reduction Visualization
    print("\n5. Generating Dimensionality Reduction Plots...")

    # PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    plt.figure(figsize=(12, 5))

    # PCA Plot
    plt.subplot(1, 2, 1)
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')
    plt.title('PCA Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.subplot(1, 2, 2)
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=clusters, cmap='viridis')
    plt.title('t-SNE Visualization')
    plt.xlabel('First t-SNE Component')
    plt.ylabel('Second t-SNE Component')

    plt.tight_layout()
    plt.show()

    # 6. Similarity Analysis
    print("\n6. Similarity Analysis:")
    similarity_matrix = cosine_similarity(features)
    avg_similarity = np.mean(similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)])
    print("Average cosine similarity between samples: {ch_score:.3f}")

    # 7. Category Analysis
    if 'Category' in df.columns:
        print("\n7. Category Distribution in Clusters:")
        category_cluster_dist = pd.DataFrame({
            'Category': df['Category'],
            'Cluster': clusters
        }).groupby(['Cluster', 'Category']).size().unstack(fill_value=0)

        print("\nCategory distribution per cluster:")
        print(category_cluster_dist)

        # Calculate cluster purity
        cluster_purity = 0
        for i in range(n_clusters):
            cluster_mask = clusters == i
            if np.any(cluster_mask):
                category_counts = df.loc[cluster_mask, 'Category'].value_counts()
                cluster_purity += category_counts.max() / np.sum(cluster_mask)
        cluster_purity /= n_clusters
        print("\nAverage cluster purity: {cluster_purity:.3f}")

    k = 5  # Set K for Precision@K and Recall@K
    y_true_relevant_items = true_labels  # Or another list of relevant items based on your system
    y_pred_top_k = predicted_labels  # Or the top K recommended items (based on your model)

    # Precision@K and Recall@K with k=5
    precision_k = precision_at_k(y_true_relevant_items, y_pred_top_k, k)
    recall_k = recall_at_k(y_true_relevant_items, y_pred_top_k, k)

    print(f"\nPrecision@{k}: {precision_k:.3f}")
    print(f"Recall@{k}: {recall_k:.3f}")

    # RMSE Example
    y_true_ratings = np.random.rand(len(true_labels))  # Replace with actual ratings if available
    y_pred_ratings = np.random.rand(len(true_labels))  # Replace with predicted ratings if available

    rmse_value = rmse(y_true_ratings, y_pred_ratings)
    print(f"RMSE: {rmse_value:.3f}")    

    return {
        'silhouette_score': sil_score,
        'calinski_harabasz_score': ch_score,
        'sparsity': sparsity,
        'avg_similarity': avg_similarity,
        'cluster_purity': cluster_purity if 'Category' in df.columns else None,
        'precision': precision if 'Category' in df.columns else None,
        'recall': recall if 'Category' in df.columns else None,
        'f1_score': f1 if 'Category' in df.columns else None,
        'accuracy': accuracy if 'Category' in df.columns else None
    }

# Run evaluation function
def run_evaluation(df, features, feature_names):
    """Run the evaluation with the processed features"""
    print("Starting evaluation...")
    print("Shape of features before cleaning:", features.shape)

    # Check for NaN values
    print("Number of NaN values:", np.isnan(features).sum())

    # Run the evaluation
    metrics = evaluate_feature_representation(features, feature_names, df)

    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 40)
    for metric, value in metrics.items():
        if value is not None:
            print("{metric}: {value:.3f}")

    return metrics

# Run the evaluation
metrics = run_evaluation(df, features, feature_names)

df.columns

from typing import Tuple, Dict, Optional


class NutritionRecommender:
    def __init__(self, food_database_df: pd.DataFrame):
        """Initialize the recommender with a food database."""
        self.df = food_database_df
        self.goals = {
            1: "Weight Loss",
            2: "Muscle Gain",
            3: "Maintenance",
            4: "Endurance Training",
            5: "Fat Loss",
            6: "Customized Goal"
        }
        
    def get_user_input(self) -> Tuple[float, float, int, str, str, bool]:
        """Get and validate user input with error handling."""
        print("Welcome to Nutrition-Based Recommendation System!")
        
        try:
            # Get and validate weight
            while True:
                try:
                    weight = float(input("Enter your weight (kg): "))
                    if 20 <= weight <= 300:  # Reasonable weight range
                        break
                    print("Please enter a realistic weight between 20 and 300 kg.")
                except ValueError:
                    print("Please enter a valid number for weight.")
            
            # Get and validate height
            while True:
                try:
                    height = float(input("Enter your height (cm): "))
                    if 100 <= height <= 250:  # Reasonable height range
                        break
                    print("Please enter a realistic height between 100 and 250 cm.")
                except ValueError:
                    print("Please enter a valid number for height.")
            
            # Get and validate age
            while True:
                try:
                    age = int(input("Enter your age: "))
                    if 12 <= age <= 120:  # Reasonable age range
                        break
                    print("Please enter a realistic age between 12 and 120 years.")
                except ValueError:
                    print("Please enter a valid number for age.")
            
            # Get and validate gender
            while True:
                gender = input("Enter your gender (male/female): ").strip().lower()
                if gender in ['male', 'female']:
                    break
                print("Please enter either 'male' or 'female'.")
            
            # Display goals menu
            print("\nChoose your fitness goal:")
            for key, value in self.goals.items():
                print(f"{key}. {value}")
            
            # Get and validate goal choice
            while True:
                try:
                    goal_choice = int(input("Enter the number corresponding to your goal: "))
                    if goal_choice in self.goals:
                        break
                    print(f"Please enter a number between 1 and {len(self.goals)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            goal = self.goals[goal_choice]
            
            # Get diabetes status
            while True:
                diabetes_input = input("\nDo you have diabetes? (yes/no): ").strip().lower()
                if diabetes_input in ['yes', 'no']:
                    has_diabetes = diabetes_input == 'yes'
                    break
                print("Please enter either 'yes' or 'no'.")
            
            return weight, height, age, gender, goal, has_diabetes
            
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
            raise
            
    def calculate_bmr(self, weight: float, height: float, age: int, gender: str) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation."""
        try:
            if gender == 'male':
                bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
            else:
                bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
            return round(bmr, 2)
        except Exception as e:
            raise ValueError(f"Error calculating BMR: {str(e)}")
    
    def get_custom_preferences(self) -> Dict[str, int]:
        """Get custom nutritional preferences from user."""
        preferences = {}
        try:
            preferences['protein_min'] = int(input("Enter minimum protein (g): "))
            preferences['carbs_max'] = int(input("Enter maximum carbs (g): "))
            preferences['fat_max'] = int(input("Enter maximum fat (g): "))
            return preferences
        except ValueError:
            raise ValueError("Please enter valid numbers for nutritional values.")
    
    def recommend_food(self, goal: str, has_diabetes: bool = False) -> pd.DataFrame:
        """Recommend foods based on user's goal and health conditions with category diversity."""
        try:
            # Adjusted filtering conditions with more realistic ranges
            conditions = {
                "Weight Loss": {
                    'protein_min': 10,  # Lowered from 15
                    'carbs_max': 35,    # Increased from 30
                    'fat_max': 15       # Increased from 10
                },
                "Muscle Gain": {
                    'protein_min': 15,  # Lowered from 20
                    'carbs_min': 15,    # Lowered from 20
                    'fat_max': 25       # Increased from 20
                },
                "Endurance Training": {
                # Modified requirements for better compatibility with diabetes
                'protein_min': 8,
                'carbs_min': 10,  # Lowered significantly
                'carbs_max': 25,  # Added maximum
                'fat_min': 5,     # Added minimum fat for energy
                'fat_max': 25
                },
                "Fat Loss": {
                    'protein_min': 15,  # Lowered from 20
                    'carbs_max': 35,    # Increased from 30
                    'fat_max': 15       # Increased from 10
                },
                "Maintenance": {
                    'protein_min': 8,   # Lowered from 10
                    'carbs_min': 8,     # Lowered from 10
                    'fat_min': 3        # Lowered from 5
                }
            }
            
            if goal == "Customized Goal":
                prefs = self.get_custom_preferences()
                filtered_df = self.df[
                    (self.df['Protein'] >= prefs['protein_min']) &
                    (self.df['Carbs'] <= prefs['carbs_max']) &
                    (self.df['Fat'] <= prefs['fat_max'])
                ]
            else:
                cond = conditions[goal]
                base_df = self.df.copy()
                
                # Apply filters based on conditions
                mask = pd.Series(True, index=base_df.index)
                if 'protein_min' in cond:
                    mask &= base_df['Protein'] >= cond['protein_min']
                if 'carbs_min' in cond:
                    mask &= base_df['Carbs'] >= cond['carbs_min']
                if 'carbs_max' in cond:
                    mask &= base_df['Carbs'] <= cond['carbs_max']
                if 'fat_min' in cond:
                    mask &= base_df['Fat'] >= cond['fat_min']
                if 'fat_max' in cond:
                    mask &= base_df['Fat'] <= cond['fat_max']
                
                filtered_df = base_df[mask]
            
            if has_diabetes:
                filtered_df = filtered_df[filtered_df['Carbs'] < 20]

            
            
            if filtered_df.empty:
                print("No foods match the strict criteria. Relaxing constraints...")
                # Reduce all minimum requirements by 30% and increase maximum limits by 30%
                for key, value in cond.items():
                    if key.endswith('_min'):
                        cond[key] = value * 0.7
                    elif key.endswith('_max'):
                        cond[key] = value * 1.3
                return self.recommend_food(goal, has_diabetes)
            
            # Calculate relevance score based on goal
            if goal in ["Weight Loss", "Fat Loss"]:
                filtered_df['relevance_score'] = (
                    filtered_df['Protein'] * 2 +
                    filtered_df['Fiber'] * 1.5 -
                    filtered_df['Fat'] * 0.5
                )
            elif goal == "Muscle Gain":
                filtered_df['relevance_score'] = (
                    filtered_df['Protein'] * 2 +
                    filtered_df['Calories'] * 0.3 +
                    filtered_df['Carbs'] * 0.5
                )
            elif goal == "Endurance Training":
                filtered_df['relevance_score'] = (
                    filtered_df['Carbs'] * 1.5 +
                    filtered_df['Protein'] * 0.8 +
                    filtered_df['Fiber'] * 0.7
                )
            else:  # Maintenance
                filtered_df['relevance_score'] = (
                    filtered_df['Protein'] +
                    filtered_df['Fiber'] +
                    filtered_df['Carbs'] * 0.5
                )
            
            # Get top recommendations with category diversity
            recommendations = []
            categories_seen = set()
            sorted_df = filtered_df.sort_values('relevance_score', ascending=False)
            
            # Try to get at least one item from each category, up to 5 total items
            for _, row in sorted_df.iterrows():
                if len(recommendations) >= 5:
                    break
                    
                if row['Category'].lower() not in categories_seen:
                    recommendations.append(row)
                    categories_seen.add(row['Category'].lower())
                elif len(recommendations) < 5 and len([r for r in recommendations if r['Category'] == row['Category']]) < 2:
                    # Allow up to 2 items from the same category if we haven't filled our recommendations
                    recommendations.append(row)
            
            recommendations_df = pd.DataFrame(recommendations)
            
            # Select relevant columns for display
            display_columns = ['Food', 'Measure', 'Grams', 'Calories', 'Protein', 
                             'Fat', 'Sat.Fat', 'Fiber', 'Carbs', 'Category']
            
            final_recommendations = recommendations_df[display_columns]
            
            return final_recommendations
            
        except Exception as e:
            raise ValueError(f"Error generating recommendations: {str(e)}")
        
    def run(self):
        """Main execution flow."""
        try:
            # Get user input
            weight, height, age, gender, goal, has_diabetes = self.get_user_input()
            
            # Calculate BMR
            bmr = self.calculate_bmr(weight, height, age, gender)
            print(f"\nYour estimated BMR is {bmr:.2f} calories/day.")
            
            # Calculate recommended daily calories based on goal
            calorie_adjustments = {
                "Weight Loss": -500,
                "Muscle Gain": 300,
                "Maintenance": 0,
                "Endurance Training": 300,
                "Fat Loss": -500,
                "Customized Goal": 0
            }
            
            recommended_calories = bmr + calorie_adjustments[goal]
            print(f"Recommended daily calories: {recommended_calories:.2f}")
            
            # Get and display recommendations
            print(f"\nBased on your goal '{goal}', we recommend the following food items:\n")
            recommendations = self.recommend_food(goal, has_diabetes)
            
            # Format and display recommendations
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(recommendations.to_string(index=False))
            
            return recommendations
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Assuming df is your food database DataFrame
    # df = pd.read_csv('food_database.csv')
    recommender = NutritionRecommender(df)
    recommender.run()


from flask import Flask, render_template, request
import pandas as pd
from typing import Dict, List, Optional

class NutritionRecommender:
    def __init__(self, food_database_df: pd.DataFrame):
        """Initialize the recommender with a food database."""
        self.df = food_database_df
        self.goals = {
            "Weight Loss": {
                'protein_min': 10,
                'carbs_max': 35,
                'fat_max': 15
            },
            "Muscle Gain": {
                'protein_min': 15,
                'carbs_min': 15,
                'fat_max': 25
            },
            "Endurance Training": {
                'protein_min': 8,
                'carbs_min': 30,
                'fat_max': 25
            },
            "Fat Loss": {
                'protein_min': 15,
                'carbs_max': 35,
                'fat_max': 15
            },
            "Maintenance": {
                'protein_min': 8,
                'carbs_min': 8,
                'fat_min': 3
            }
        }

    def calculate_bmr(self, weight: float, height: float, age: int, gender: str) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation."""
        try:
            if gender.lower() == 'male':
                bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
            else:
                bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
            return round(bmr, 2)
        except Exception as e:
            raise ValueError(f"Error calculating BMR: {str(e)}")

    def calculate_daily_calories(self, bmr: float, goal: str) -> float:
        """Calculate recommended daily calories based on goal."""
        calorie_adjustments = {
            "Weight Loss": -500,
            "Muscle Gain": 300,
            "Maintenance": 0,
            "Endurance Training": 300,
            "Fat Loss": -500,
            "Customized Goal": 0
        }
        return bmr + calorie_adjustments.get(goal, 0)

    def recommend_food(self, goal: str, bmr: float, weight: float, 
                      has_diabetes: bool = False, 
                      custom_prefs: Optional[Dict] = None) -> pd.DataFrame:
        """Recommend foods based on user's goal and health conditions."""
        try:
            # Base filtering conditions
            if goal == "Customized Goal" and custom_prefs:
                filtered_df = self.df[
                    (self.df['Protein'] >= custom_prefs['protein_min']) &
                    (self.df['Carbs'] <= custom_prefs['carbs_max']) &
                    (self.df['Fat'] <= custom_prefs['fat_max'])
                ]
            else:
                # Get calorie target
                daily_calories = self.calculate_daily_calories(bmr, goal)
                meal_calories = daily_calories / 3  # Assuming 3 meals per day
                
                base_df = self.df.copy()
                
                # Apply goal-specific filters
                if goal in self.goals:
                    conditions = self.goals[goal]
                    mask = pd.Series(True, index=base_df.index)
                    
                    # Apply protein requirements based on weight
                    protein_per_kg = {
                        "Weight Loss": 1.2,
                        "Muscle Gain": 1.6,
                        "Endurance Training": 1.2,
                        "Fat Loss": 1.4,
                        "Maintenance": 1.0
                    }
                    
                    min_protein = weight * protein_per_kg.get(goal, 1.0)
                    mask &= base_df['Protein'] >= conditions.get('protein_min', min_protein)
                    
                    # Apply other nutrient conditions
                    if 'carbs_min' in conditions:
                        mask &= base_df['Carbs'] >= conditions['carbs_min']
                    if 'carbs_max' in conditions:
                        mask &= base_df['Carbs'] <= conditions['carbs_max']
                    if 'fat_min' in conditions:
                        mask &= base_df['Fat'] >= conditions['fat_min']
                    if 'fat_max' in conditions:
                        mask &= base_df['Fat'] <= conditions['fat_max']
                    
                    # Apply calorie filter
                    mask &= base_df['Calories'] <= meal_calories * 1.2  # Allow 20% flexibility
                    
                    filtered_df = base_df[mask]
            
            if has_diabetes:
                filtered_df = filtered_df[filtered_df['Carbs'] < 20]
            
            if filtered_df.empty:
                print("No foods match the strict criteria. Relaxing constraints...")
                # Implement fallback logic here
                return self.recommend_food(goal, bmr * 1.2, weight * 0.8, has_diabetes, custom_prefs)
            
            # Calculate relevance score based on goal
            if goal in ["Weight Loss", "Fat Loss"]:
                filtered_df['relevance_score'] = (
                    filtered_df['Protein'] * 2 +
                    filtered_df['Fiber'] * 1.5 -
                    filtered_df['Fat'] * 0.5
                )
            elif goal == "Muscle Gain":
                filtered_df['relevance_score'] = (
                    filtered_df['Protein'] * 2 +
                    filtered_df['Calories'] * 0.3 +
                    filtered_df['Carbs'] * 0.5
                )
            elif goal == "Endurance Training":
                filtered_df['relevance_score'] = (
                    filtered_df['Carbs'] * 1.5 +
                    filtered_df['Protein'] * 0.8 +
                    filtered_df['Fiber'] * 0.7
                )
            else:  # Maintenance
                filtered_df['relevance_score'] = (
                    filtered_df['Protein'] +
                    filtered_df['Fiber'] +
                    filtered_df['Carbs'] * 0.5
                )
            
            # Get diverse recommendations
            recommendations = []
            categories_seen = set()
            sorted_df = filtered_df.sort_values('relevance_score', ascending=False)
            
            for _, row in sorted_df.iterrows():
                if len(recommendations) >= 5:
                    break
                    
                if row['Category'].lower() not in categories_seen:
                    recommendations.append(row)
                    categories_seen.add(row['Category'].lower())
                elif len(recommendations) < 5 and len([r for r in recommendations if r['Category'] == row['Category']]) < 2:
                    recommendations.append(row)
            
            final_df = pd.DataFrame(recommendations)
            display_columns = ['Food', 'Measure', 'Grams', 'Calories', 'Protein', 
                             'Fat', 'Sat.Fat', 'Fiber', 'Carbs', 'Category']
            
            return final_df[display_columns]
            
        except Exception as e:
            raise ValueError(f"Error generating recommendations: {str(e)}")

# Create Flask app
app = Flask(__name__)

# Load your food database
# df = pd.read_csv('your_food_database.csv')
# Initialize recommender
recommender = NutritionRecommender(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get form data
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = int(request.form['age'])
        gender = request.form['gender']
        goal = request.form['goal']
        has_diabetes = request.form.get('diabetes') == 'yes'

        # Calculate BMR
        bmr = recommender.calculate_bmr(weight, height, age, gender)
        daily_calories = recommender.calculate_daily_calories(bmr, goal)

        # Handle customized goal
        custom_prefs = None
        if goal == 'Customized Goal':
            custom_prefs = {
                'protein_min': float(request.form['protein_min']),
                'carbs_max': float(request.form['carbs_max']),
                'fat_max': float(request.form['fat_max'])
            }

        # Get recommendations
        recommendations = recommender.recommend_food(
            goal=goal,
            bmr=bmr,
            weight=weight,
            has_diabetes=has_diabetes,
            custom_prefs=custom_prefs
        )

        return render_template('recommendations.html',
                             bmr=bmr,
                             daily_calories=daily_calories,
                             recommendations=recommendations.to_dict(orient='records'),
                             goal=goal)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
