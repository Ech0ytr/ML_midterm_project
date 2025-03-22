# Sales Success Playbook (DS5640 Midterm Project)

Welcome! This project is part of our midterm for the **DS5640 Machine Learning** course at Vanderbilt. Our goal is to build a smart, data-driven **sales playbook** that helps sales reps make better decisions — from the very first deal opportunity to a smooth implementation.

## 📚 What We're Working On

We're analyzing real-world CRM data (anonymized) to:
- Predict which deals are most likely to succeed
- Understand what makes implementations go smoothly
- Segment customers based on things like company size, industry, and tech stack
- Recommend sales strategies tailored to different customer profiles
- Visualize it all in an interactive dashboard!

## 🧩 Datasets

We’re using three main datasets provided by HubSpot (fully anonymized):
1. **Deals** – contains info on sales opportunities (status, value, duration, etc.)
2. **Companies** – includes company size, industry, region, technologies, etc.
3. **Tickets** – tracks implementation milestones and training activities

Each dataset gives us a different piece of the customer journey puzzle.

## 🛠️ Our Stack

- **Python (pandas, sklearn, matplotlib, seaborn)**
- **Jupyter Notebooks**
- **Logistic Regression, Tree Models, Boosting**
- **GridSearchCV for hyperparameter tuning**
- **Docker** for reproducible deployment
- **Optional**: Plotly / Dash / Streamlit for dashboard building

## 🚧 Project Milestones

- ✅ Data exploration & cleaning
- ✅ Feature engineering & merging datasets
- ⏳ Baseline model training
- ⏳ Model tuning and evaluation
- ⏳ Dashboard creation
- ⏳ Docker setup & deployment
- ⏳ Final business insights & recommendations

## 📈 What Success Looks Like

- A clear, actionable sales playbook based on real patterns in the data
- An easy-to-use dashboard for sales reps
- Models that explain *why* certain deals succeed or fail
- Business takeaways that could help improve real-world sales processes

## 📁 Repo Structure (WIP)
├── data/ # CSV files (deals, companies, tickets) ├── notebooks/ # EDA, modeling, and analysis notebooks ├── src/ # ETL and utility scripts ├── dashboard/ # Dashboard app (if applicable) ├── docker/ # Docker config & files ├── README.md # This file! └── requirements.txt # Python dependencies


Thanks for stopping by! 👋  
We'll keep updating this repo as we go. Feedback is welcome — and if you're curious, feel free to explore the notebooks.
