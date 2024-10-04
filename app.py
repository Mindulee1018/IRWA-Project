import pandas as pd
import nltk

from flask import Flask, request, render_template
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
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+mysqldb://root:@localhost:3307/ecommerce"
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



#Creating tags
# Define the set of stop words
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

# routes
@app.route("/")
def index():
    top_rated_items = pd.read_csv('top_rated_products.csv')
    return render_template('index.html', top_rated_items=top_rated_items, truncate=truncate)

@app.route("/index")
def home():   
    return render_template('index.html', truncate=truncate)

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



# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username,password=password)
        db.session.add(new_signup)
        db.session.commit()       


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        content_based_rec = content_based_recommendations(df, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."

            # Make sure to load the top rated items to show them
            top_rated_items = pd.read_csv('top_rated_products.csv')

            return render_template('index.html', message=message,  top_rated_items=top_rated_items, truncate=truncate)
        else:
            return render_template('recommendations.html', content_based_rec=content_based_rec, truncate=truncate)
        
    # If the request is not POST, just render the index
    return render_template('index.html', truncate=truncate)
       

if __name__=='__main__':
    app.run(debug=True)