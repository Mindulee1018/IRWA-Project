from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

print("App is starting...")


# load files
data_file = 'cleaned_train_data.csv'  # Adjust the path if necessary
train_data = pd.read_csv(data_file)

# Sort by rating_count and get the top 10 most-rated products
top_rated_products = train_data.nlargest(10, 'rating_count')[['product_name', 'category', 'discounted_price', 'actual_price', 'rating', 'rating_count']]

# database configuration
app.secret_key = "Mindu2002"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecommerce"
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


# Recommendations functions

def content_based_recommendations(train_data, product_name):
    # Check if the item name exists in the training data
    if product_name not in train_data['product_name'].values:
        print(f"Item '{product_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['about_product'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = train_data[train_data['product_name'] == product_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the indices of all similar items (excluding the first item, which is the product itself)
    similar_item_indices = [i[0] for i in similar_items if i[0] != item_index]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[similar_item_indices][['product_name', 'rating_count', 'rating']]

    return recommended_items_details

# routes
@app.route("/")
def index():
    
    # Pass the top 10 products to the HTML template
    return render_template('index.html', products=top_rated_products.to_dict(orient='records'))


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
        content_based_rec = content_based_recommendations(train_data, prod)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('index.html', message=message)
       

if __name__=='__main__':
    app.run(debug=True)