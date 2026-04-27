from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this for production

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['DATA_FOLDER'] = 'data/uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure directories exist
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
os.makedirs('instance', exist_ok=True)

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'

# Create tables
with app.app_context():
    db.create_all()

# Password validation
def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search("[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search("[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search("[0-9]", password):
        return False, "Password must contain at least one digit"
    if not re.search("[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

# Authentication decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'danger')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_default_dataset():
    """Load or create default dataset in the data directory"""
    default_path = os.path.join(app.config['DATA_FOLDER'], 'Mall_Customers1.csv')
    
    if not os.path.exists(default_path):
        sample_data = """CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72"""
        with open(default_path, 'w') as f:
            f.write(sample_data)
    
    return pd.read_csv(default_path)

def analyze_data(df):
    # Basic statistics with error handling
    try:
        stats = {
            'total_customers': len(df),
            'gender_dist': df['Gender'].value_counts().to_dict(),
            'age_stats': {
                'min': df['Age'].min(),
                'max': df['Age'].max(),
                'mean': round(df['Age'].mean()), 
                'median': df['Age'].median()
            },
            'income_stats': {
                'min': df['Annual Income (k$)'].min(),
                'max': df['Annual Income (k$)'].max(),
                'mean': round(df['Annual Income (k$)'].mean()),
                'median': df['Annual Income (k$)'].median()
            },
            'spending_stats': {
                'min': df['Spending Score (1-100)'].min(),
                'max': df['Spending Score (1-100)'].max(),
                'mean': round(df['Spending Score (1-100)'].mean()),
                'median': df['Spending Score (1-100)'].median()
            }
        }
    except Exception as e:
        stats = {
            'error': f"Error calculating statistics: {str(e)}",
            'total_customers': len(df) if 'df' in locals() else 0
        }
    
    # Generate visualizations with error handling
    plots = {}
    
    try:
        # Age Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Age', bins=20, kde=True)
        plt.title('Age Distribution of Customers')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['age_dist'] = base64.b64encode(buf.read()).decode('ascii')
        plt.close()
    except Exception as e:
        plots['age_dist_error'] = f"Could not generate age distribution: {str(e)}"
    
    try:
        # Income vs Spending Score
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
        plt.title('Income vs Spending Score')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['income_spending'] = base64.b64encode(buf.read()).decode('ascii')
        plt.close()
    except Exception as e:
        plots['income_spending_error'] = f"Could not generate income vs spending plot: {str(e)}"
    
    try:
        # Gender Distribution
        plt.figure(figsize=(6, 6))
        df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Gender Distribution')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['gender_dist'] = base64.b64encode(buf.read()).decode('ascii')
        plt.close()
    except Exception as e:
        plots['gender_dist_error'] = f"Could not generate gender distribution: {str(e)}"
    
    try:
        # Correlation Heatmap
        plt.figure(figsize=(8, 6))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['correlation'] = base64.b64encode(buf.read()).decode('ascii')
        plt.close()
    except Exception as e:
        plots['correlation_error'] = f"Could not generate correlation heatmap: {str(e)}"
    
    try:
        # Spending Score by Age Group
        df['Age Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 60, 70, 100], 
                                labels=['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '70+'])
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Age Group', y='Spending Score (1-100)')
        plt.title('Spending Score by Age Group')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['spending_age'] = base64.b64encode(buf.read()).decode('ascii')
        plt.close()
    except Exception as e:
        plots['spending_age_error'] = f"Could not generate spending by age group: {str(e)}"
    
    return stats, plots

def perform_clustering(X, max_clusters=10):
    results = {}
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    try:
        # Find optimal k using elbow method with safety checks
        if len(X) < 2:
            raise ValueError("Not enough samples for clustering")
        
        actual_max = min(max_clusters, len(X) - 1)
        if actual_max < 2:
            raise ValueError(f"Only {len(X)} samples available - need at least 2 for clustering")
        
        wcss = []
        for i in range(1, actual_max + 1):
            try:
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            except Exception as e:
                wcss.append(None)
                continue
        
        # Plot elbow method
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, actual_max + 1), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        results['elbow_plot'] = base64.b64encode(buf.read()).decode('ascii')
        plt.close()
        
        # Determine optimal k (simple heuristic - look for the "elbow")
        optimal_k = min(5, actual_max)  # Default to 5 or actual_max if smaller
        
        # Apply K-means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        
        # Plot clusters
        plt.figure(figsize=(12, 8))
        for i in range(optimal_k):
            plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, label=f'Cluster {i+1}')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   s=300, c='yellow', label='Centroids')
        plt.title('Customer Segments')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        results['cluster_plot'] = base64.b64encode(buf.read()).decode('ascii')
        plt.close()
        
        results['clusters'] = {
            'optimal_k': optimal_k,
            'cluster_sizes': pd.Series(y_kmeans).value_counts().to_dict(),
            'success': True
        }
        
    except Exception as e:
        results['error'] = f"Clustering failed: {str(e)}"
        results['clusters'] = {
            'success': False,
            'error_message': str(e)
        }
    
    return results

@app.route('/')
@login_required
def index():
    try:
        df = get_default_dataset()
        stats, plots = analyze_data(df)
        return render_template('index.html', stats=stats, plots=plots, 
                             data=df.head(10).to_dict('records'))
    except Exception as e:
        flash(f"Error loading dataset: {str(e)}", 'danger')
        return render_template('index.html', stats={}, plots={}, data=[])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password, password):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        session['user_id'] = user.id
        session['username'] = user.username
        flash('Login successful', 'success')
        
        next_page = request.args.get('next')
        return redirect(next_page) if next_page else redirect(url_for('index'))
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not password or not confirm_password:
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        is_valid, message = validate_password(password)
        if not is_valid:
            flash(message, 'danger')
            return redirect(url_for('register'))
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# @app.route('/upload', methods=['GET', 'POST'])
# @login_required
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No file part', 'danger')
#             return redirect(request.url)
        
#         file = request.files['file']
#         if file.filename == '':
#             flash('No selected file', 'danger')
#             return redirect(request.url)
        
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['DATA_FOLDER'], filename)
#             try:
#                 file.save(filepath)
#                 df = pd.read_csv(filepath)
                
#                 # Validate required columns
#                 required_columns = ['CustomerID', 'Gender', 'Age', 
#                                   'Annual Income (k$)', 'Spending Score (1-100)']
#                 if not all(col in df.columns for col in required_columns):
#                     missing = [col for col in required_columns if col not in df.columns]
#                     raise ValueError(f"Missing required columns: {', '.join(missing)}")
                
#                 stats, plots = analyze_data(df)
#                 return render_template('report.html', stats=stats, plots=plots, 
#                                     data=df.head(10).to_dict('records'), filename=filename)
#             except Exception as e:
#                 if os.path.exists(filepath):
#                     os.remove(filepath)
#                 return render_template('upload.html', error=str(e))
    
#     return render_template('upload.html')




@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = "Mall_Customers1.csv"  # <- Renamed file
            filepath = os.path.join(app.config['DATA_FOLDER'], filename)
            try:
                file.save(filepath)
                df = pd.read_csv(filepath)
                
                # Validate required columns
                required_columns = ['CustomerID', 'Gender', 'Age', 
                                  'Annual Income (k$)', 'Spending Score (1-100)']
                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    raise ValueError(f"Missing required columns: {', '.join(missing)}")
                
                stats, plots = analyze_data(df)
                return render_template('report.html', stats=stats, plots=plots, 
                                    data=df.head(10).to_dict('records'), filename=filename)
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return render_template('upload.html', error=str(e))
    
    return render_template('upload.html')

@app.route('/report')
@login_required
def report():
    try:
        df = get_default_dataset()
        stats, plots = analyze_data(df)
        
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        cluster_results = perform_clustering(X)
        
        if 'clusters' in cluster_results:
            stats['clusters'] = cluster_results['clusters']
        
        return render_template('report.html', 
                            stats=stats, 
                            plots=plots, 
                            data=df.head(10).to_dict('records'),
                            elbow_plot=cluster_results.get('elbow_plot'),
                            cluster_plot=cluster_results.get('cluster_plot'),
                            cluster_error=cluster_results.get('error'))
    except Exception as e:
        flash(f"Error generating report: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/data/<filename>')
@login_required
def download_file(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)





