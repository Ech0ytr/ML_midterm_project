import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings
import json
import pickle
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import traceback


# Suppress warnings
warnings.filterwarnings("ignore")

class SalesPlaybookModel:
    def __init__(self, deals_path, companies_path, tickets_path, mappings_path=None):
        self.deals_path = deals_path
        self.companies_path = companies_path
        self.tickets_path = tickets_path
        self.mappings_path = mappings_path
        
        # Instance variables for data
        self.deals = None
        self.companies = None
        self.tickets = None
        self.merged_df = None
        self.mappings = None
        
        # Model components
        self.classifier = None
        self.regressor = None
        self.preprocessing_pipeline = None
        self.cluster_model = None
        
        # Load the data
        self.load_data()
        
    def load_data(self):
        """Load datasets and mappings"""
        print("Loading datasets...")
        
        # Load CSV files
        self.deals = pd.read_csv(self.deals_path)
        self.companies = pd.read_csv(self.companies_path)
        self.tickets = pd.read_csv(self.tickets_path)
        
        print(f"Loaded deals: {len(self.deals)} records")
        print(f"Loaded companies: {len(self.companies)} records")
        print(f"Loaded tickets: {len(self.tickets)} records")
        
        # Load mappings if provided
        if self.mappings_path:
            with open(self.mappings_path, 'r') as f:
                self.mappings = json.load(f)
            print("Loaded mappings data")
    
    def process_and_merge_data(self):
        """Clean, transform and merge datasets"""
        print("Processing and merging data...")
        
        # ----- PREPARE COMPANIES DATA -----
        # Filter out columns with more than 80% missing values
        valid_company_cols = self.companies.columns[self.companies.isnull().mean() < 0.8].tolist()
        company_subset = self.companies[valid_company_cols]
        
        # Ensure ID columns are string type for consistent merging
        if 'Record ID' in company_subset.columns:
            company_subset['Record ID'] = company_subset['Record ID'].astype(str)
        
        # ----- PREPARE DEALS DATA -----
        # Convert date columns to datetime
        date_columns = [col for col in self.deals.columns if 'Date' in col]
        for col in date_columns:
            if col in self.deals.columns:
                self.deals[col] = pd.to_datetime(self.deals[col], errors='coerce')
        
        # Create time_to_close_days as target variable
        if 'Create Date' in self.deals.columns and 'Close Date' in self.deals.columns:
            self.deals['time_to_close_days'] = (self.deals['Close Date'] - self.deals['Create Date']).dt.days
            
            # Filter out invalid or extreme values
            self.deals = self.deals.dropna(subset=['time_to_close_days'])
            self.deals = self.deals[(self.deals['time_to_close_days'] > 0) & (self.deals['time_to_close_days'] <= 365)]
        
        # Ensure deal ID columns are string type
        if 'Record ID' in self.deals.columns:
            self.deals['Record ID'] = self.deals['Record ID'].astype(str)
        
        # ----- MERGE TICKETS WITH DEALS -----
        # Count tickets per deal if possible
        if 'Associated Deal' in self.tickets.columns and 'Record ID' in self.deals.columns:
            # Create mapping for tickets to deals
            ticket_counts = self.tickets['Associated Deal'].value_counts().reset_index()
            ticket_counts.columns = ['Record ID', 'ticket_count_per_deal']
            
            # Convert to string for consistent merging
            ticket_counts['Record ID'] = ticket_counts['Record ID'].astype(str)
            
            # Merge ticket counts with deals
            self.deals = self.deals.merge(ticket_counts, on='Record ID', how='left')
            self.deals['ticket_count_per_deal'] = self.deals['ticket_count_per_deal'].fillna(0)
        else:
            # Create a default column if no mapping is possible
            self.deals['ticket_count_per_deal'] = 0
            
        # ----- PROCESS TRAINING FLAGS -----
        # Extract training data from tickets
        if 'Training: General Overview' in self.tickets.columns:
            training_columns = [col for col in self.tickets.columns if 'Training:' in col]
            
            # Create a flag for any training provided
            if len(training_columns) > 0:
                self.tickets['Any Training Provided'] = self.tickets[training_columns].notna().any(axis=1)
                
                # Create mapping to merge with deals
                if 'Associated Company (Primary)' in self.tickets.columns:
                    training_per_company = self.tickets[['Associated Company (Primary)', 'Any Training Provided']].copy()
                    training_per_company = training_per_company.drop_duplicates('Associated Company (Primary)')
                    
                    # Merge training data if possible
                    if 'Associated Company (Primary)' in self.deals.columns:
                        self.deals = self.deals.merge(
                            training_per_company, 
                            on='Associated Company (Primary)', 
                            how='left'
                        )
                        self.deals['Any Training Provided'] = self.deals['Any Training Provided'].fillna(False)
        
        # ----- MERGE DEALS WITH COMPANIES -----
        # Check if we have mappings to use
        if self.mappings and 'CompanyToDeals' in self.mappings:
            # Create a mapping dictionary: Deal ID -> Company ID
            deal_to_company = {}
            for company_id, deal_ids in self.mappings['CompanyToDeals'].items():
                for deal_id in deal_ids:
                    deal_to_company[deal_id] = company_id
            
            # Add company ID to each deal based on mapping
            self.deals['Company Record ID'] = self.deals['Record ID'].map(deal_to_company)
            
            # Merge deals with companies
            self.merged_df = self.deals.merge(
                company_subset,
                left_on='Company Record ID',
                right_on='Record ID',
                how='left',
                suffixes=('_deal', '_company')
            )
        else:
            # Direct merge on Associated Company if available
            if 'Associated Company (Primary)' in self.deals.columns and 'Company name' in company_subset.columns:
                self.merged_df = self.deals.merge(
                    company_subset,
                    left_on='Associated Company (Primary)',
                    right_on='Company name',
                    how='left'
                )
            else:
                # If no proper join is possible, just use the deals
                self.merged_df = self.deals.copy()
                print("Warning: Could not properly merge deals with companies.")
        
        # ----- FEATURE ENGINEERING -----
        # Create time-based features
        # Ensure Create Date is datetime
            self.deals['Create Date'] = pd.to_datetime(self.deals['Create Date'], errors='coerce')

            # Create time-based features
            self.deals['Create Month'] = self.deals['Create Date'].dt.month
            self.deals['Create Quarter'] = self.deals['Create Date'].dt.quarter
            self.deals['Create Year'] = self.deals['Create Date'].dt.year
            self.deals['Days Since Creation'] = (pd.Timestamp.today() - self.deals['Create Date']).dt.days

        if 'Create Date' in self.merged_df.columns:
            self.merged_df['Create Month'] = self.merged_df['Create Date'].dt.month
            self.merged_df['Create Quarter'] = self.merged_df['Create Date'].dt.quarter
            self.merged_df['Create Year'] = self.merged_df['Create Date'].dt.year
            
            # Days since creation
            latest_date = self.merged_df['Create Date'].max()
            self.merged_df['Days Since Creation'] = (latest_date - self.merged_df['Create Date']).dt.days
        
        # Create deal duration categories
        if 'time_to_close_days' in self.merged_df.columns:
            self.merged_df['Duration Category'] = pd.cut(
                self.merged_df['time_to_close_days'],
                bins=[0, 30, 90, 180, 365],
                labels=['Short', 'Medium', 'Long', 'Very Long']
            )
        
        # Create win/loss target feature
        if 'Is Closed Won' in self.merged_df.columns:
            self.merged_df['Closed_Won_Label'] = self.merged_df['Is Closed Won'].map({True: 'Closed Won', False: 'Closed Lost'})
        elif 'Deal Stage' in self.merged_df.columns:
            self.merged_df['Closed_Won_Label'] = self.merged_df['Deal Stage'].apply(
                lambda x: 'Closed Won' if 'Won' in str(x) else 'Closed Lost'
            )
        
        print(f"Processed and merged data: {len(self.merged_df)} records with {len(self.merged_df.columns)} features")
        return self.merged_df
    
    def perform_exploratory_analysis(self, output_dir=None):
        """Perform exploratory analysis and generate insights"""
        print("Performing exploratory analysis...")
        
        if self.merged_df is None:
            print("No merged data available. Call process_and_merge_data() first.")
            return
        
        # Create output directory if provided
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        # ----- ANALYZE MISSING VALUES -----
        plt.figure(figsize=(12, 6))
        missing_counts = self.merged_df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) > 0:
            missing_counts.head(20).plot(kind='bar')
            plt.title('Missing Values Per Column')
            plt.ylabel('Number of Missing Values')
            plt.xlabel('Columns')
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(f"{output_dir}/missing_values.png")
                plt.close()
            else:
                plt.show()
        
        # ----- ANALYZE WIN RATES -----
        if 'Closed_Won_Label' in self.merged_df.columns:
            # Win rate by ticket volume
            if 'ticket_count_per_deal' in self.merged_df.columns:
                # Define ticket volume cohort buckets
                def bucket_ticket_count(n):
                    if n == 0:
                        return "0 Tickets"
                    elif n <= 2:
                        return "1-2 Tickets"
                    else:
                        return "3+ Tickets"
                
                self.merged_df['Ticket Volume Cohort'] = self.merged_df['ticket_count_per_deal'].apply(bucket_ticket_count)
                
                # Analyze success rate by ticket volume cohort
                ticket_cohort_props = (
                    self.merged_df.groupby('Ticket Volume Cohort')['Closed_Won_Label']
                    .value_counts(normalize=True)
                    .unstack()
                    .fillna(0)
                )
                
                plt.figure(figsize=(8, 5))
                ticket_cohort_props.plot(kind='bar', stacked=True)
                plt.title('Deal Success Rate by Ticket Volume Cohort')
                plt.ylabel('Proportion')
                plt.xlabel('Ticket Volume Cohort')
                plt.xticks(rotation=0)
                plt.legend(loc='upper right')
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(f"{output_dir}/success_by_ticket_volume.png")
                    plt.close()
                else:
                    plt.show()
            
            # Win rate by training exposure
            if 'Any Training Provided' in self.merged_df.columns:
                cohort_props = (
                    self.merged_df.groupby('Any Training Provided')['Closed_Won_Label']
                    .value_counts(normalize=True)
                    .unstack()
                    .fillna(0)
                )
                
                plt.figure(figsize=(8, 5))
                cohort_props.plot(kind='bar', stacked=True)
                plt.title('Deal Success Rate by Training Exposure')
                plt.ylabel('Proportion')
                plt.xlabel('Any Training Provided')
                plt.xticks([0, 1], ['No Training', 'Training Provided'], rotation=0)
                plt.legend(loc='upper right')
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(f"{output_dir}/success_by_training.png")
                    plt.close()
                else:
                    plt.show()
        
        # ----- ANALYZE TIME TO CLOSE -----
        if 'time_to_close_days' in self.merged_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.merged_df['time_to_close_days'], bins=30)
            plt.title('Distribution of Time to Close (Days)')
            plt.xlabel('Days to Close')
            plt.ylabel('Count')
            
            if output_dir:
                plt.savefig(f"{output_dir}/time_to_close_distribution.png")
                plt.close()
            else:
                plt.show()
            
            # Time to close by deal outcome
            if 'Closed_Won_Label' in self.merged_df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='Closed_Won_Label', y='time_to_close_days', data=self.merged_df)
                plt.title('Time to Close by Deal Outcome')
                plt.ylabel('Days to Close')
                plt.xlabel('Deal Outcome')
                
                if output_dir:
                    plt.savefig(f"{output_dir}/time_to_close_by_outcome.png")
                    plt.close()
                else:
                    plt.show()
        
        return "Completed exploratory analysis"
    
    def cluster_companies(self, n_clusters=5):
        print(f"Clustering companies into {n_clusters} segments...")
        
        if self.merged_df is None or len(self.merged_df) == 0:
            print("No merged data available. Call process_and_merge_data() first.")
            return None
            
        # Make a copy to avoid modifying the original
        cluster_df = self.merged_df.copy()
        
        # Create time-based features if they don't exist
        if 'Create Date' in cluster_df.columns:
            try:
                # Convert to datetime first
                cluster_df['Create Date'] = pd.to_datetime(cluster_df['Create Date'], errors='coerce')
                
                # Add derived date columns
                cluster_df['Create Month'] = cluster_df['Create Date'].dt.month
                cluster_df['Create Quarter'] = cluster_df['Create Date'].dt.quarter
                cluster_df['Create Year'] = cluster_df['Create Date'].dt.year
                
                # Calculate days since creation using the current date
                latest_date = cluster_df['Create Date'].max()
                cluster_df['Days Since Creation'] = (latest_date - cluster_df['Create Date']).dt.days
            except Exception as e:
                print(f"Warning: Could not process date columns: {e}")
        
        # Define potential clustering features
        potential_categorical = ['Deal Stage', 'Forecast category', 'Duration Category',
                                'Pipeline', 'Industry', 'Company Size']
        potential_numerical = ['Amount', 'Deal probability', 'Weighted amount', 'Forecast amount',
                            'Create Month', 'Create Quarter', 'Create Year', 'Days Since Creation',
                            'Number of Employees', 'Annual Revenue', 'ticket_count_per_deal']
        
        # Filter to only use columns that actually exist in the dataframe
        categorical = [col for col in potential_categorical if col in cluster_df.columns]
        numerical = [col for col in potential_numerical if col in cluster_df.columns]
        
        # Make sure we have at least some features to work with
        if len(categorical) == 0 and len(numerical) == 0:
            print("Error: No valid clustering features found in the dataset")
            return None
        
        print(f"Using categorical features: {categorical}")
        print(f"Using numerical features: {numerical}")
        
        # Clean data for clustering (drop rows with missing values in selected features)
        selected_features = categorical + numerical
        analysis_df = cluster_df[selected_features].copy()
        
        # Handle missing values for numerical columns
        for col in numerical:
            analysis_df[col] = analysis_df[col].fillna(analysis_df[col].median())
        
        # Handle categorical features by converting to string and filling missing values
        # This is key to fix the categorical error
        for col in categorical:
            # Convert to string first to avoid categorical issues
            analysis_df[col] = analysis_df[col].astype(str)
            # Replace 'nan' strings with 'Unknown'
            analysis_df[col] = analysis_df[col].replace('nan', 'Unknown')
        
        # Setup preprocessing with scaling and one-hot encoding
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.cluster import KMeans
        from sklearn.pipeline import Pipeline
        
        try:
            # Create preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical),
                    ('cat', categorical_transformer, categorical)
                ]
            )
            
            # Apply preprocessing
            X_cluster_ready = preprocessor.fit_transform(analysis_df)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_cluster_ready)
            
            # Store the cluster model and preprocessor
            self.cluster_model = kmeans
            self.preprocessing_pipeline = preprocessor
            
            # Add cluster labels back to analysis dataframe
            analysis_df['Cluster'] = cluster_labels
            self.analysis_df = analysis_df  # Store for later use
            
            # Create a summary of cluster characteristics
            # For numeric features, use mean
            if numerical:
                cluster_summary = analysis_df.groupby('Cluster')[numerical].mean().round(2)
            else:
                # Create empty DataFrame if no numerical features
                cluster_summary = pd.DataFrame(index=range(n_clusters))
                cluster_summary.index.name = 'Cluster'
            
            # Add categorical feature distributions
            for cat_col in categorical:
                # Get distribution of categories within each cluster
                cat_dist = pd.crosstab(
                    analysis_df['Cluster'],
                    analysis_df[cat_col],
                    normalize='index'
                ).round(2)
                
                # Find the dominant category for each cluster
                if not cat_dist.empty:
                    dominant_cats = cat_dist.idxmax(axis=1).to_frame(f'Top {cat_col}')
                    
                    # Add to summary
                    cluster_summary = pd.merge(
                        cluster_summary,
                        dominant_cats,
                        left_index=True,
                        right_index=True,
                    )
            
            # Add count of companies in each cluster
            cluster_counts = analysis_df['Cluster'].value_counts().sort_index().to_frame('Company Count')
            cluster_summary = pd.merge(
                cluster_summary,
                cluster_counts,
                left_index=True,
                right_index=True,
            )
            
            # Calculate cluster success metrics if available
            if 'Is Closed Won' in cluster_df.columns:
                # Create a mapping from analysis_df to cluster_df
                cluster_mapping = analysis_df['Cluster'].to_dict()
                
                # Add cluster labels to original dataframe where possible
                cluster_df.loc[analysis_df.index, 'Cluster'] = cluster_df.loc[analysis_df.index].index.map(cluster_mapping)
                
                # Calculate win rate by cluster
                win_rate = cluster_df.groupby('Cluster')['Is Closed Won'].mean().round(3) * 100
                win_rate = win_rate.to_frame('Win Rate %')
                
                # Add to summary
                cluster_summary = pd.merge(
                    cluster_summary,
                    win_rate,
                    left_index=True,
                    right_index=True,
                    how='left'  # Use left join in case some clusters have no win/loss data
                )
            
            # If time to close days is available, add average per cluster
            if 'time_to_close_days' in cluster_df.columns:
                avg_time = cluster_df.groupby('Cluster')['time_to_close_days'].mean().round(1)
                avg_time = avg_time.to_frame('Avg Days to Close')
                
                # Add to summary
                cluster_summary = pd.merge(
                    cluster_summary,
                    avg_time,
                    left_index=True,
                    right_index=True,
                    how='left'  # Use left join in case some clusters have no time data
                )
                
            print("Clustering completed successfully")
            return cluster_summary
            
        except Exception as e:
            print(f"Error in clustering process: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_clusters(self, n_clusters=5):
        # Add these imports at the top of the function
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'analysis_df') or self.analysis_df is None or 'Cluster' not in self.analysis_df.columns:
            return {"error": "No clustering data available. Run clustering first."}
        
        result = {}
        
        try:
            # Check which columns are available for visualization
            # Remove date columns from visualization if they don't exist in the analysis_df
            date_columns = ['Create Month', 'Create Quarter', 'Create Year', 'Days Since Creation']
            
            # Get all numeric columns except Cluster for PCA
            numeric_cols = [col for col in self.analysis_df.columns 
                        if col != 'Cluster' 
                        and pd.api.types.is_numeric_dtype(self.analysis_df[col])]
            
            # Remove any date columns that aren't in the dataframe
            numeric_cols = [col for col in numeric_cols if col not in date_columns or col in self.analysis_df.columns]
            
            if len(numeric_cols) >= 2:  # Need at least 2 dimensions for PCA
                # Create PCA for visualization
                pca = PCA(n_components=2)
                
                # Create scatter plot of clusters
                plt.figure(figsize=(10, 8))
                
                # Apply PCA to numeric features
                pca_result = pca.fit_transform(self.analysis_df[numeric_cols])
                
                # Create scatter plot
                scatter = plt.scatter(
                    pca_result[:, 0], 
                    pca_result[:, 1], 
                    c=self.analysis_df['Cluster'], 
                    cmap='viridis', 
                    alpha=0.6, 
                    s=50
                )
                
                # Add cluster centers if we have the model
                if self.cluster_model is not None:
                    # Transform cluster centers through PCA
                    centers_pca = pca.transform(self.preprocessing_pipeline.named_transformers_['num'].inverse_transform(
                        self.cluster_model.cluster_centers_[:, :len(numeric_cols)]
                    ))
                    
                    # Add cluster centers to plot
                    plt.scatter(
                        centers_pca[:, 0], 
                        centers_pca[:, 1], 
                        c=range(n_clusters), 
                        cmap='viridis', 
                        marker='x', 
                        s=200, 
                        linewidths=3
                    )
                
                # Add labels and legend
                plt.title('Customer Segments (PCA Projection)', fontsize=14)
                plt.xlabel(f'Principal Component 1', fontsize=12)
                plt.ylabel(f'Principal Component 2', fontsize=12)
                plt.colorbar(scatter, label='Cluster')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add variance explained
                explained_variance = pca.explained_variance_ratio_
                plt.figtext(
                    0.02, 0.02, 
                    f'Explained variance: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f}',
                    fontsize=10
                )
                
                result['scatter_plot'] = plt.gcf()
                plt.close()
                
            else:
                result['error'] = "Insufficient numeric data for visualization"
                
            return result
            
        except Exception as e:
            return {"error": f"Error performing clustering visualization: {str(e)}"}
        
    def train_deal_outcome_model(self):
        """Train a model to predict deal outcome (win/loss)"""
        if self.merged_df is None or len(self.merged_df) == 0:
            print("No data available for training")
            return None
        
        # Make a copy of the data for training
        train_df = self.merged_df.copy()
        
        # Ensure we have the target variable
        if 'Is Closed Won' not in train_df.columns:
            print("Error: Target variable 'Is Closed Won' not found in data")
            return None
        
        # Process date columns before training
        if 'Create Date' in train_df.columns:
            try:
                # Convert to datetime
                train_df['Create Date'] = pd.to_datetime(train_df['Create Date'], errors='coerce')
                
                # Explicitly add date-derived columns
                train_df['Create Month'] = train_df['Create Date'].dt.month
                train_df['Create Quarter'] = train_df['Create Date'].dt.quarter
                train_df['Create Year'] = train_df['Create Date'].dt.year
                
                # Calculate days since creation
                latest_date = train_df['Create Date'].max()
                train_df['Days Since Creation'] = (latest_date - train_df['Create Date']).dt.days
            except Exception as e:
                print(f"Warning: Could not process date columns: {e}")
        
        # Define features for training
        categorical_features = ['Deal Stage', 'Forecast category', 'Pipeline', 'Industry', 'Company Size']
        numerical_features = ['Amount', 'Deal probability', 'Weighted amount', 'Forecast amount',
                            'Number of Employees', 'Annual Revenue', 'ticket_count_per_deal']
        
        # Add date features if they exist
        date_features = ['Create Month', 'Create Quarter', 'Create Year', 'Days Since Creation']
        for feature in date_features:
            if feature in train_df.columns:
                numerical_features.append(feature)
        
        # Filter to columns that exist in the dataframe
        categorical_features = [col for col in categorical_features if col in train_df.columns]
        numerical_features = [col for col in numerical_features if col in train_df.columns]
        
        # Combine all features
        all_features = categorical_features + numerical_features
        
        if not all_features:
            print("Error: No valid features for training")
            return None
        
        # Handle missing values
        for col in numerical_features:
            train_df[col] = train_df[col].fillna(train_df[col].median())
        
        for col in categorical_features:
            train_df[col] = train_df[col].astype(str).fillna('Unknown')
        
        # Define X and y
        X = train_df[all_features]
        y = train_df['Is Closed Won']
        
        # Create preprocessing pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        model.fit(X, y)
        
        # Store the model
        self.classifier = model
        
        # Calculate model performance
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = model.predict(X)
        
        results = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred, average='weighted')
        }
        
        print(f"Deal outcome model trained: Accuracy = {results['accuracy']:.2f}, F1 = {results['f1_score']:.2f}")
        return results

    def train_time_to_close_model(self):
        """Train a model to predict time to close for deals"""
        # Import numpy within the function to ensure it's available
        import numpy as np
        
        if self.merged_df is None or len(self.merged_df) == 0:
            print("No data available for training")
            return None
        
        # Make a copy of the data for training
        train_df = self.merged_df.copy()
        
        # Ensure we have the target variable
        if 'time_to_close_days' not in train_df.columns:
            print("Error: Target variable 'time_to_close_days' not found in data")
            return None
        
        # Process date columns before training
        if 'Create Date' in train_df.columns:
            try:
                # Convert to datetime
                train_df['Create Date'] = pd.to_datetime(train_df['Create Date'], errors='coerce')
                
                # Explicitly add date-derived columns
                train_df['Create Month'] = train_df['Create Date'].dt.month
                train_df['Create Quarter'] = train_df['Create Date'].dt.quarter
                train_df['Create Year'] = train_df['Create Date'].dt.year
                
                # Calculate days since creation
                latest_date = train_df['Create Date'].max()
                train_df['Days Since Creation'] = (latest_date - train_df['Create Date']).dt.days
            except Exception as e:
                print(f"Warning: Could not process date columns: {e}")
        
        # Define features for training
        categorical_features = ['Deal Stage', 'Forecast category', 'Pipeline', 'Industry', 'Company Size']
        numerical_features = ['Amount', 'Deal probability', 'Weighted amount', 'Forecast amount',
                            'Number of Employees', 'Annual Revenue', 'ticket_count_per_deal']
        
        # Add date features if they exist
        date_features = ['Create Month', 'Create Quarter', 'Create Year', 'Days Since Creation']
        for feature in date_features:
            if feature in train_df.columns:
                numerical_features.append(feature)
        
        # Filter to columns that exist in the dataframe
        categorical_features = [col for col in categorical_features if col in train_df.columns]
        numerical_features = [col for col in numerical_features if col in train_df.columns]
        
        # Combine all features
        all_features = categorical_features + numerical_features
        
        if not all_features:
            print("Error: No valid features for training")
            return None
        
        # Handle missing values
        for col in numerical_features:
            train_df[col] = train_df[col].fillna(train_df[col].median())
        
        for col in categorical_features:
            train_df[col] = train_df[col].astype(str).fillna('Unknown')
        
        # Define X and y - logarithmic transformation of target for better regression
        X = train_df[all_features]
        y = np.log1p(train_df['time_to_close_days'])  # Log transform to normalize
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        model.fit(X, y)
        
        # Store the model
        self.regressor = model
        
        # Calculate model performance
        y_pred = model.predict(X)
        
        results = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2_score': r2_score(y, y_pred)
        }
        
        print(f"Time to close model trained: RMSE = {results['rmse']:.2f}, R² = {results['r2_score']:.2f}")
        return results
    
    def predict_new_deal(self, deal_data):
        import numpy as np
        import pandas as pd
        from datetime import datetime
        
        if self.classifier is None or self.regressor is None:
            print("Models not trained. Call train_deal_outcome_model() and train_time_to_close_model() first.")
            return None
        
        # Create a DataFrame from the input data
        deal_df = pd.DataFrame([deal_data])
        
        # Process the input similar to training data
        # Generate features using same transformations as training
        if 'Create Date' in deal_df.columns:
            try:
                deal_df['Create Date'] = pd.to_datetime(deal_df['Create Date'])
                deal_df['Create Month'] = deal_df['Create Date'].dt.month
                deal_df['Create Quarter'] = deal_df['Create Date'].dt.quarter
                deal_df['Create Year'] = deal_df['Create Date'].dt.year
                deal_df['Days Since Creation'] = (datetime.now() - deal_df['Create Date']).dt.days
            except Exception as e:
                print(f"Warning: Could not process date features: {e}")
        
        # Add default values for missing columns
        expected_columns = [
            'Amount', 'Deal probability', 'Weighted amount', 'Forecast amount',
            'Create Date', 'Pipeline', 'Deal Stage', 'Forecast category',
            'Industry', 'Number of Employees', 'Annual Revenue', 'ticket_count_per_deal'
        ]
        
        # Add any missing columns with default values
        for col in expected_columns:
            if col not in deal_df.columns:
                if col in ['Amount', 'Deal probability', 'Weighted amount', 'Forecast amount', 
                        'Number of Employees', 'Annual Revenue', 'ticket_count_per_deal']:
                    deal_df[col] = 0  # Default numeric value
                else:
                    deal_df[col] = 'Unknown'  # Default categorical value
        
        # Make predictions
        try:
            win_prob = self.classifier.predict_proba(deal_df)[:, 1][0]
        except Exception as e:
            print(f"Error predicting win probability: {e}")
            win_prob = 0.5  # Default value
        
        try:
            close_time_log = self.regressor.predict(deal_df)[0]
            close_time_days = np.expm1(close_time_log)
        except Exception as e:
            print(f"Error predicting closing time: {e}")
            close_time_days = 30  # Default value
        
        return {
            'win_probability': win_prob,
            'estimated_days_to_close': close_time_days,
            'estimated_close_date': (datetime.now() + pd.Timedelta(days=close_time_days)).strftime('%Y-%m-%d')
        }
    
    def get_feature_importances(self, model_type='classifier'):
        if model_type == 'classifier' and self.classifier is None:
            return None
        if model_type == 'regressor' and self.regressor is None:
            return None
        
        try:
            if model_type == 'classifier':
                model = self.classifier
            else:
                model = self.regressor
            
            # Check if the model is a Pipeline
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                estimator = model.named_steps['classifier']
            elif hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
                estimator = model.named_steps['regressor']
            else:
                estimator = model
            
            # Get feature names
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                
                # Get transformed feature names
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    # If get_feature_names_out is not available, use generic feature names
                    feature_names = [f'feature_{i}' for i in range(len(estimator.feature_importances_))]
            else:
                feature_names = [f'feature_{i}' for i in range(len(estimator.feature_importances_))]
            
            # Get feature importances
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
            else:
                return None
            
            # Create DataFrame
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
            
            return feature_importance_df
        
        except Exception as e:
            print(f"Error getting feature importances: {e}")
            return None

    def save_models(self, output_dir="models"):
        import os
        import pickle
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(self, 'classifier') and self.classifier:
            with open(f"{output_dir}/outcome_classifier.pkl", 'wb') as f:
                pickle.dump(self.classifier, f)
        
        if hasattr(self, 'regressor') and self.regressor:
            with open(f"{output_dir}/time_regressor.pkl", 'wb') as f:
                pickle.dump(self.regressor, f)
        
        if hasattr(self, 'cluster_model') and self.cluster_model:
            with open(f"{output_dir}/cluster_model.pkl", 'wb') as f:
                pickle.dump(self.cluster_model, f)
        
        if hasattr(self, 'preprocessing_pipeline') and self.preprocessing_pipeline:
            with open(f"{output_dir}/preprocessing_pipeline.pkl", 'wb') as f:
                pickle.dump(self.preprocessing_pipeline, f)
        
        print(f"Models saved to {output_dir}")
        return True

    def load_models(self, input_dir="models"):
        import os
        import pickle
        
        try:
            # Check if directory exists
            if not os.path.exists(input_dir):
                print(f"Models directory '{input_dir}' does not exist")
                return False
                
            # Try to load the classifier
            classifier_path = f"{input_dir}/outcome_classifier.pkl"
            if os.path.exists(classifier_path):
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
            
            # Try to load the regressor
            regressor_path = f"{input_dir}/time_regressor.pkl"
            if os.path.exists(regressor_path):
                with open(regressor_path, 'rb') as f:
                    self.regressor = pickle.load(f)
            
            # Try to load the cluster model
            cluster_path = f"{input_dir}/cluster_model.pkl"
            if os.path.exists(cluster_path):
                with open(cluster_path, 'rb') as f:
                    self.cluster_model = pickle.load(f)
            
            # Try to load the preprocessing pipeline
            pipeline_path = f"{input_dir}/preprocessing_pipeline.pkl"
            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'rb') as f:
                    self.preprocessing_pipeline = pickle.load(f)
            
            print(f"Models loaded from {input_dir}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = SalesPlaybookModel(
        deals_path="anonymized_hubspot_deals.csv",
        companies_path="anonymized_hubspot_companies.csv",
        tickets_path="anonymized_hubspot_tickets.csv",
        mappings_path="mappings.json"
    )
    
    # Process and merge the data
    model.process_and_merge_data()
    
    # Perform exploratory analysis
    model.perform_exploratory_analysis(output_dir="analysis_output")
    
    # Cluster companies
    model.cluster_companies(n_clusters=5)
    
    # Train predictive models
    model.train_deal_outcome_model()
    model.train_time_to_close_model()
    
    # Save the trained models
    model.save_models(output_dir="models")
    
    # Example prediction for a new deal
    sample_deal = {
        'Amount': 50000,
        'Deal probability': 0.7,
        'Weighted amount': 35000,
        'Forecast amount': 40000,
        'Create Date': '2025-01-15',
        'Pipeline': 'Sales Pipeline',
        'Deal Stage': 'Negotiation',
        'Forecast category': 'Best Case',
        'ticket_count_per_deal': 2
    }
    
    prediction = model.predict_new_deal(sample_deal)
    print("\nSample Deal Prediction:")
    print(f"Win Probability: {prediction['win_probability']:.2%}")
    print(f"Estimated Days to Close: {prediction['estimated_days_to_close']:.1f}")
    print(f"Estimated Close Date: {prediction['estimated_close_date']}")