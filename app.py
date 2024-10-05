import pandas as pd
import nltk

from flask import Flask, request, render_template, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download punkt 
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# load files
top_rated_products = pd.read_csv("top_rated_products.csv")
df = pd.read_csv("cleaned_train_data.csv")

# database configuration
app.secret_key = "Mindu2002"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root@localhost/ecommerce"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define model class for browsing history
class BrowsingHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    product_id = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    product_names = db.Column(db.String(500), nullable=True)

# Preprocess text for tags extraction
stop_words = set(stopwords.words('english'))

def clean_and_extract_tags(text):
    # Tokenize the text and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Filter out non-alphanumeric tokens and stop words
    tags = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ', '.join(tags)

columns_to_extract_tags_from = ['category', 'about_product']

for column in columns_to_extract_tags_from:
    df[column] = df[column].apply(clean_and_extract_tags)

# Concatenate the cleaned tags from all relevant columns
df['Tags'] = df[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)


# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

# Recommendations functions

#Rating based recommendation
average_ratings = df.groupby(['product_id', 'product_name', 'actual_price', 'category', 'about_product', 'product_link', 'rating_count'])['rating'].mean().reset_index()
top_rated_items = average_ratings.sort_values(by='rating', ascending=False)

rating_base_recommendation = top_rated_items.head(10)
rating_base_recommendation['rating'] = rating_base_recommendation['rating'].astype(int)
rating_base_recommendation['rating_count'] = rating_base_recommendation['rating_count'].astype(int)

#Contenct Based Recommendation System
def content_based_recommendations(df, item_name, top_n=10):
    # Normalize the item name to remove any leading/trailing spaces and make it case insensitive
    item_name_cleaned = item_name.strip().lower()

    # Check if the item name exists in the training data using a case-insensitive search
    matching_items = df[df['product_name'].str.contains(item_name_cleaned, case=False, na=False)]

    # Debugging information
    print(f"Searching for: '{item_name_cleaned}'")
    print(f"Matching items found: {matching_items.shape[0]}")

    if matching_items.empty:
        print(f"Item '{item_name_cleaned}' not found in the training data.")
        return pd.DataFrame()  # Return an empty DataFrame

    # If there are matching items, get the index of the first match
    item_index = matching_items.index[0]

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Get the cosine similarity scores for the matched item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n + 1]

    # Check if there are enough similar items to recommend
    if not top_similar_items:
        return pd.DataFrame()  # Return an empty DataFrame if no similar items are found

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = df.iloc[recommended_item_indices][['product_name', 'rating_count', 'rating']]

    return recommended_items_details

# Collaborative Filtering Recommendations (Stub)
def collaborative_filtering_recommendations(df, target_user_id, top_n=10):
    # Create a user-item matrix
    user_item_matrix = df.pivot_table(values='rating', index='user_id', columns='product_id')
    
    # Fill NaN values with 0 for collaborative filtering
    user_item_matrix = user_item_matrix.fillna(0)

    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix)

    # Create a DataFrame for easier user similarity handling
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Get the similar users for the target user
    similar_users = user_similarity_df[target_user_id].sort_values(ascending=False).index[1:top_n + 1]

    # Get product recommendations from similar users
    recommendations = pd.Series(dtype=float)
    
    for similar_user in similar_users:
        similar_user_products = user_item_matrix.loc[similar_user]
        recommendations = pd.concat([recommendations, similar_user_products[similar_user_products > 0]])

    # Aggregate recommendations and filter top N products
    recommendations = recommendations.groupby(recommendations.index).sum().sort_values(ascending=False).head(top_n)

    return df[df['product_id'].isin(recommendations.index)][['product_name', 'rating_count', 'rating']]

# Hybrid Recommendations
def get_browsing_history(user_id):
    history = BrowsingHistory.query.filter_by(user_id=user_id).all()
    return [entry.product_id for entry in history]

def hybrid_recommendations(df, target_user_id, item_name, top_n=10):
    content_based_rec = content_based_recommendations(df, item_name, top_n)
    collaborative_filtering_rec = collaborative_filtering_recommendations(df, target_user_id, top_n)
    browsinghistory = get_browsing_history(target_user_id)

    # Combine recommendations while ensuring no duplicates
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec])

    if browsinghistory:
        history_recommendations = df[df['product_id'].isin(browsinghistory)].head(top_n)
        hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec, history_recommendations]).drop_duplicates()

    return hybrid_rec.head(top_n)

# routes
@app.route("/")
def index():
    target_user_id = session.get('user_id')
    item_name = ""
    if target_user_id is None:
        # Redirect to the login page or show an appropriate message
        return render_template('index.html', top_rated_items=rating_base_recommendation, truncate=truncate, message="Please log in to see personalized recommendations.")
    
    hybrid_rec = hybrid_recommendations(df, target_user_id, item_name, top_n=10)

    return render_template('index.html', top_rated_items=rating_base_recommendation, truncate=truncate, hybrid_rec=hybrid_rec)

@app.route("/index")
def home():   
    return render_template('index.html', truncate=truncate, top_rated_items=rating_base_recommendation)

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        return redirect(url_for('index'))



# Route for sigin page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        
        user = Signin.query.filter_by(username=username, password=password).first()

        if user:
            # Store the user_id in the session
            session['user_id'] = user.id
            
            # Render the index page
            return redirect(url_for('index'))
        else:
            # If credentials are incorrect, show an error message
            return render_template('index.html', error="Invalid credentials")

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user_id from session
    return render_template('signin.html', message="You have been logged out.")

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        target_user_id = session.get('user_id')
        item_name = request.form.get('prod')
        nbr = int(request.form.get('nbr'))

        if target_user_id and item_name:
            # Create a new BrowsingHistory entry
            history_entry = BrowsingHistory(user_id=target_user_id, product_id=None, timestamp=db.func.current_timestamp(), product_names=item_name)  # product_id is None for search
            db.session.add(history_entry)
            db.session.commit()

        content_based_rec = content_based_recommendations(df, item_name, top_n=nbr)

        if content_based_rec.empty:
            message = "no recommendations available for this product."
            
            return render_template('index.html', top_rated_items=rating_base_recommendation, truncate=truncate, message=message)
        else:
            return render_template('recommendations.html', content_based_rec = content_based_rec, truncate=truncate)
    return render_template('index.html', top_rated_items=rating_base_recommendation, truncate=truncate)

if __name__=='__main__':
    app.run(debug=True)