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
from datetime import datetime

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
    signin_time = db.Column(default=datetime.now)

    def __init__(self, user_id, product_name):
        self.user_id = user_id
        self.product_name = product_name

# Define model class for browsing history
class Browsinghistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    product_name = db.Column(db.String(100), nullable=False)

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
    recommended_items_details = df.iloc[recommended_item_indices][['product_id', 'product_name', 'actual_price', 'rating_count', 'rating']]

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

# Function to get user's most recent search products
def get_user_recent_searches(user_id, df, limit=10):
    
    # Query the BrowsingHistory for the current user's searches ordered by most recent
    recent_searches = (Browsinghistory.query
                       .filter_by(user_id=user_id)
                       .order_by(Browsinghistory.id.desc())
                       .limit(limit)
                       .all())
    
    # Create a list to hold detailed product information
    product_details = []
    seen_products = set()  # To track seen product names

    for search in recent_searches:
        # Search for the product details in the DataFrame
        product_detail = df[df['product_name'] == search.product_name]
        
        if not product_detail.empty:
            # Extract the relevant details and append to the list
            product_info = {
                'product_id': product_detail['product_id'].values[0],
                'product_name': product_detail['product_name'].values[0],
                'actual_price': product_detail['actual_price'].values[0],
                'rating': product_detail['rating'].values[0]
            }
            # Check if the product has already been seen
            if product_info['product_name'] not in seen_products:
                product_details.append(product_info)  # Add to the list
                seen_products.add(product_info['product_name'])  # Mark as seen

    return product_details

# routes
@app.route("/", methods=['GET'])
def index():
    # Check if user is logged in
    recent_searches = []
    if 'user_id' in session:
        user_id = session['user_id']  # Get the logged-in user's ID
        
        # Get the top 10 most recent searches by the user
        recent_searches = get_user_recent_searches(user_id, df, limit=10)
    
    # Load top-rated items (assuming 'rating_base_recommendation' is a list of top-rated products)
    top_rated_items = rating_base_recommendation  # Your existing recommendation logic
    
    # Render the index page with both top-rated items and recent searches
    return render_template('index.html', 
                           top_rated_items=top_rated_items, 
                           recent_searches=recent_searches, 
                           truncate=truncate)


@app.route("/index")
def home():   
        # Check if user is logged in
    recent_searches = []
    if 'user_id' in session:
        user_id = session['user_id']  # Get the logged-in user's ID
        
        # Get the top 10 most recent searches by the user
        recent_searches = get_user_recent_searches(user_id, df, limit=10)
    
    # Load top-rated items (assuming 'rating_base_recommendation' is a list of top-rated products)
    top_rated_items = rating_base_recommendation  # Your existing recommendation logic
    
    # Render the index page with both top-rated items and recent searches
    return render_template('index.html', 
                           top_rated_items=top_rated_items, 
                           recent_searches=recent_searches, 
                           truncate=truncate)

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
        
        user = Signup.query.filter_by(username=username).first()

        if user:
            # Verify the password
            if user.password == password:
                # Store the user_id in the session
                session['user_id'] = user.id

                signin_entry = Signin(username=user.username)
                db.session.add(signin_entry)
                db.session.commit()
            
                # Render the index page
                return redirect(url_for('index', truncate=truncate, top_rated_items=rating_base_recommendation))
            
            else:
                # If credentials are incorrect, show an error message
                return render_template('index', error="Invalid credentials")
        else:
            # Username does not exist
            return render_template('index.html', error="Username not found")
        

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user_id from session
    return render_template('signin.html', message="You have been logged out.")

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')  # Get the product searched for recommendations
        nbr = int(request.form.get('nbr'))  # Get the number of recommendations

        # Check if user is logged in
        if 'user_id' in session:
            user_id = session['user_id']  # Get the logged-in user's ID

            # Log the search to BrowsingHistory
            browsing_history_entry = Browsinghistory(user_id=user_id, product_name=prod)
            db.session.add(browsing_history_entry)
            db.session.commit()

        # Generate content-based recommendations
        content_based_rec = content_based_recommendations(df, prod, top_n=nbr)

        # Handle the case where no recommendations are found
        if content_based_rec.empty:
            message = "No recommendations available for this product."

            return render_template('index.html', message=message, top_rated_items=rating_base_recommendation, truncate=truncate)
        else:
            return render_template('recommendations.html', content_based_rec=content_based_rec, truncate=truncate)

    # If the request is not POST, just render the index page
    return render_template('index.html', truncate=truncate)




if __name__=='__main__':
    app.run(debug=True)