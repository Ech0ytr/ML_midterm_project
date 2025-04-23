Deal Prediction & Customer Segmentation
This project presents a complete machine learning pipeline to predict B2B sales outcomes, estimate time-to-close, and segment customers into meaningful groups. It combines exploratory data analysis, model training, clustering, and deployment into an interactive Streamlit dashboard, containerized with Docker for portability.

# Project Goals
Predict whether a deal will be won or lost

Estimate the number of days it will take to close a deal

Cluster deals into customer segments for strategic targeting

Deliver these insights through an interactive dashboard

# Dataset Description
The project integrates three datasets exported from HubSpot:

Companies: Firmographic data including industry, revenue, and employee count

Deals: Deal progression data with timestamps, values, and outcomes

Tickets: Support interaction data including training, project milestones, and touchpoints

Data was joined using a custom mappings.json structure and cleaned to handle missing values and inconsistent formatting.

# Features Engineered
time_to_close_days: Target for regression, based on creation and close dates

Temporal features: Create Month, Create Quarter, Days Since Creation

Support interaction count: ticket_count_per_deal

Duration categories for clustering: Short, Medium, Long, Very Long

# Modeling Approach
Two main tasks were tackled:

## 1. Classification: Deal Outcome
Goal: Predict if a deal will be “Closed Won”

Model: Random Forest Classifier

Features: Company size, revenue, deal value, support activity, and time-based features

Metrics: Accuracy and F1-score

## 2. Regression: Time to Close
Goal: Estimate how long a deal takes to close

Model: Random Forest Regressor

Features: Same as above

Metrics: RMSE and R²

## 3. Clustering: Customer Segmentation
Goal: Identify key customer personas

Model: KMeans with one-hot encoding and PCA visualization

Segments: Includes Fast Closers, Strategic Buyers, and Low Engagement Leads

# Application Interface
A user-friendly Streamlit dashboard allows:

Previewing and uploading datasets

Training models dynamically

Viewing classification and regression metrics

Visualizing cluster personas

Making predictions for new deals

# Deployment with Docker
The project is fully containerized using Docker.

## To build and run:
bash
Copy
Edit
docker-compose up --build
Access the dashboard at:
http://localhost:8501

# Project Structure
bash
Copy
Edit
.
├── app.py                     # Streamlit frontend
├── sales_playbook_model.py   # Core model class with clustering & ML
├── EDA.ipynb                 # Data exploration and validation
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/                     # Folder to store input CSVs
├── models/                   # Folder to store trained models

# Team
Echo Yu – yut10

Ziyi Tao – taoz4

Linxuan Fan – fanl5

# References
Scikit-learn Documentation

Streamlit Documentation

HubSpot Developer Docs

Docker Documentation

Pandas

NumPy

Matplotlib & Seaborn
