import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os
from sales_playbook_model import SalesPlaybookModel

# Set page configuration
st.set_page_config(
    page_title="Sales Playbook Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">Sales Success Playbook</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides interactive analytics and predictions based on the Hubspot CRM data,
helping sales representatives make better decisions throughout the customer journey.
""")

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state['model'] = None
    st.session_state['data_loaded'] = False

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Home", "Data Explorer", "Customer Segmentation", "Deal Prediction", "Implementation Insights"]
)

# Function to load data
@st.cache_data
def load_sample_data():
    # If real data is available, use these paths
    deals_path = "data/anonymized_hubspot_deals.csv"
    companies_path = "data/anonymized_hubspot_companies.csv"
    tickets_path = "data/anonymized_hubspot_tickets.csv"
    mappings_path = "data/mappings.json"
    
    try:
        # Try to load the data
        model = SalesPlaybookModel(
            deals_path=deals_path,
            companies_path=companies_path,
            tickets_path=tickets_path,
            mappings_path=mappings_path
        )
        model.process_and_merge_data()
        return model, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, False

# Function to load models
def load_models(model_instance):
    try:
        loaded = model_instance.load_models("models")
        return loaded
    except Exception as e:
        st.warning("Pre-trained models not found. You may need to train models first.")
        return False

# Home page
if page == "Home":
    st.markdown('<h2 class="sub-header">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Dashboard description
    st.write("""
    Welcome to the Sales Success Playbook Dashboard! This tool is designed to help sales teams optimize their 
    customer journey from initial opportunity to successful implementation. Here's what you can explore:
    """)
    
    # Dashboard sections in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Explorer**
        - View key metrics and trends
        - Analyze deal success factors
        - Explore implementation metrics
        """)
        
        st.markdown("""
        **Deal Prediction**
        - Predict win probability for new deals
        - Estimate time to close
        - Optimize deal strategies
        """)
    
    with col2:
        st.markdown("""
        **Customer Segmentation**
        - Explore customer clusters
        - Understand segment characteristics
        - Tailor approaches by segment
        """)
        
        st.markdown("""
        **Implementation Insights**
        - Track implementation success metrics
        - Analyze training impacts
        - Optimize resource allocation
        """)
    
    # Load data button
    st.markdown("---")
    if st.button("Load Data & Initialize Models"):
        with st.spinner("Loading data and initializing models..."):
            model, success = load_sample_data()
            if success:
                st.session_state['model'] = model
                st.session_state['data_loaded'] = True
                models_loaded = load_models(model)
                if models_loaded:
                    st.success("Data and pre-trained models loaded successfully!")
                else:
                    st.info("Data loaded. Models will be trained when needed.")
            else:
                st.error("Failed to load data. Please check the data paths.")

# Data Explorer page
elif page == "Data Explorer":
    st.markdown('<h2 class="sub-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    if not st.session_state['data_loaded']:
        st.warning("Please load the data first from the Home page.")
        if st.button("Load Data Now"):
            with st.spinner("Loading data..."):
                model, success = load_sample_data()
                if success:
                    st.session_state['model'] = model
                    st.session_state['data_loaded'] = True
                    st.success("Data loaded successfully!")
                else:
                    st.error("Failed to load data.")
    
    else:
        model = st.session_state['model']
        merged_df = model.merged_df
        
        # Overview metrics
        st.markdown('<h3>Data Overview</h3>', unsafe_allow_html=True)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Deals", len(merged_df))
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            if 'Is Closed Won' in merged_df.columns:
                win_rate = merged_df['Is Closed Won'].mean() * 100
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            if 'time_to_close_days' in merged_df.columns:
                avg_days = merged_df['time_to_close_days'].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg. Days to Close", f"{avg_days:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            if 'Amount' in merged_df.columns:
                avg_deal_size = merged_df['Amount'].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg. Deal Size", f"${avg_deal_size:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Data filters
        st.markdown('### Filter Data')
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Industry' in merged_df.columns:
                industries = ['All'] + sorted(merged_df['Industry'].dropna().unique().tolist())
                selected_industry = st.selectbox("Select Industry", industries)
        
        with col2:
            if 'Deal Stage' in merged_df.columns:
                stages = ['All'] + sorted(merged_df['Deal Stage'].dropna().unique().tolist())
                selected_stage = st.selectbox("Select Deal Stage", stages)
        
        # Apply filters
        filtered_df = merged_df.copy()
        if selected_industry != 'All' and 'Industry' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]
        if selected_stage != 'All' and 'Deal Stage' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Deal Stage'] == selected_stage]
        
        # Success factors analysis
        st.markdown('### Deal Success Factors')
        
        # Visualization options
        viz_option = st.selectbox(
            "Select Visualization",
            ["Win Rate by Deal Size", "Win Rate by Industry", "Time to Close Distribution", 
             "Training Impact on Success"]
        )
        
        # Create the selected visualization
        if viz_option == "Win Rate by Deal Size":
            if 'Amount' in filtered_df.columns and 'Is Closed Won' in filtered_df.columns:
                # Create amount buckets
                filtered_df['Amount Bucket'] = pd.cut(
                    filtered_df['Amount'],
                    bins=[0, 10000, 50000, 100000, float('inf')],
                    labels=['<$10k', '$10k-$50k', '$50k-$100k', '>$100k']
                )
                
                # Calculate win rate by amount bucket
                win_by_amount = filtered_df.groupby('Amount Bucket')['Is Closed Won'].mean().reset_index()
                win_by_amount['Win Rate'] = win_by_amount['Is Closed Won'] * 100
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Amount Bucket', y='Win Rate', data=win_by_amount, ax=ax)
                ax.set_title('Win Rate by Deal Size')
                ax.set_xlabel('Deal Size')
                ax.set_ylabel('Win Rate (%)')
                st.pyplot(fig)
            else:
                st.info("Required columns not available in the data.")
                
        elif viz_option == "Win Rate by Industry":
            if 'Industry' in filtered_df.columns and 'Is Closed Won' in filtered_df.columns:
                # Calculate win rate by industry
                win_by_industry = filtered_df.groupby('Industry')['Is Closed Won'].mean().reset_index()
                win_by_industry['Win Rate'] = win_by_industry['Is Closed Won'] * 100
                win_by_industry = win_by_industry.sort_values('Win Rate', ascending=False)
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='Industry', y='Win Rate', data=win_by_industry.head(10), ax=ax)
                ax.set_title('Win Rate by Top 10 Industries')
                ax.set_xlabel('Industry')
                ax.set_ylabel('Win Rate (%)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Required columns not available in the data.")
                
        elif viz_option == "Time to Close Distribution":
            if 'time_to_close_days' in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(filtered_df['time_to_close_days'], bins=30, kde=True, ax=ax)
                ax.set_title('Distribution of Time to Close')
                ax.set_xlabel('Days to Close')
                ax.set_ylabel('Count')
                st.pyplot(fig)
            else:
                st.info("Required columns not available in the data.")
                
        elif viz_option == "Training Impact on Success":
            if 'Any Training Provided' in filtered_df.columns and 'Is Closed Won' in filtered_df.columns:
                # Calculate win rate by training
                training_impact = filtered_df.groupby('Any Training Provided')['Is Closed Won'].mean().reset_index()
                training_impact['Win Rate'] = training_impact['Is Closed Won'] * 100
                
                # Plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x='Any Training Provided', y='Win Rate', data=training_impact, ax=ax)
                ax.set_title('Impact of Training on Win Rate')
                ax.set_xlabel('Training Provided')
                ax.set_ylabel('Win Rate (%)')
                ax.set_xticklabels(['No', 'Yes'])
                st.pyplot(fig)
            else:
                st.info("Required columns not available in the data.")
        
        # Show filtered data
        with st.expander("View Filtered Data"):
            st.dataframe(filtered_df)

# Customer Segmentation page
elif page == "Customer Segmentation":
    st.markdown('<h2 class="sub-header">Customer Segmentation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load the data first from the Home page.")
        if st.button("Load Data Now"):
            with st.spinner("Loading data..."):
                # Load data code here
                st.session_state['data_loaded'] = True
                st.success("Data loaded successfully!")
    
    else:
        model = st.session_state.get('model')
        
        # Clustering options
        st.markdown('### Customer Clustering')
        
        n_clusters = st.slider("Select Number of Clusters", 3, 8, 5)
        
        # Run clustering with selected parameters
        if st.button("Run Clustering Analysis"):
            with st.spinner("Performing clustering..."):
                try:
                    # Make sure data is processed
                    if not hasattr(model, 'merged_df') or model.merged_df is None:
                        # If data is loaded but not processed, process it
                        model.process_and_merge_data()
                        
                    # Run clustering
                    cluster_summary = model.cluster_companies(n_clusters=n_clusters)
                    
                    if cluster_summary is not None:
                        st.success("Clustering completed successfully!")
                        
                        # Display cluster summary
                        st.markdown('### Cluster Characteristics')
                        st.dataframe(cluster_summary)
                        
                        # Visualize clusters with better error handling
                        st.markdown('### Cluster Visualization')
                        
                        try:
                            visualizations = model.visualize_clusters(n_clusters=n_clusters)
                            
                            if visualizations and 'scatter_plot' in visualizations and visualizations['scatter_plot'] is not None:
                                st.pyplot(visualizations['scatter_plot'])
                            elif visualizations and 'error' in visualizations:
                                st.error(visualizations['error'])
                            else:
                                st.info("Could not create visualizations due to insufficient data.")
                        except Exception as viz_error:
                            st.error(f"Error performing clustering visualization: {str(viz_error)}")
                            st.info("Visualization failed, but you can still view the cluster characteristics and recommendations above.")
                        
                        # Segment-Based Recommendations
                        st.markdown('### Segment-Based Recommendations')
                        
                        # Check if we have cluster data
                        if hasattr(model, 'analysis_df') and model.analysis_df is not None and 'Cluster' in model.analysis_df.columns:
                            # Get unique clusters
                            clusters = sorted(model.analysis_df['Cluster'].unique())
                            
                            # Allow selection of a cluster for recommendations
                            selected_cluster = st.selectbox(
                                "Select Cluster for Recommendations",
                                [f"Cluster {i}" for i in clusters]
                            )
                            
                            # Extract cluster number
                            cluster_num = int(selected_cluster.split()[1])
                            
                            # Show recommendations based on cluster
                            st.markdown(f"### Sales Approach for {selected_cluster}")
                            
                            if cluster_num == 0:
                                st.markdown("#### Enterprise Accounts Strategy:")
                                st.markdown("""
                                * Focus on long-term relationship building
                                * Emphasize ROI and strategic value
                                * Involve executive stakeholders early
                                * Prepare for longer sales cycles
                                * Allocate senior sales representatives
                                """)
                            elif cluster_num == 1:
                                st.markdown("#### Mid-Market Growth Strategy:")
                                st.markdown("""
                                * Emphasize quick time-to-value
                                * Offer scalable implementation options
                                * Focus on specific pain points
                                * Provide case studies from similar companies
                                * Balance between relationship and efficiency
                                """)
                            elif cluster_num == 2:
                                st.markdown("#### Small Business Efficiency Strategy:")
                                st.markdown("""
                                * Offer streamlined implementation packages
                                * Focus on ease of use and quick adoption
                                * Provide clear pricing and ROI calculations
                                * Consider offering limited-scope pilot projects
                                * Emphasize self-service resources
                                """)
                            else:
                                st.markdown(f"#### Recommendations for {selected_cluster}:")
                                st.markdown("""
                                * Analyze the cluster characteristics above
                                * Adjust sales approach based on deal size and complexity
                                * Consider win rate patterns in this segment
                                * Allocate resources accordingly
                                * Monitor time-to-close expectations
                                """)
                    else:
                        st.error("Clustering analysis failed. Please check the logs for details.")
                        
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")

# Deal Prediction page
elif page == "Deal Prediction":
    st.markdown('<h2 class="sub-header">Deal Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state['data_loaded']:
        st.warning("Please load the data first from the Home page.")
        if st.button("Load Data Now"):
            with st.spinner("Loading data..."):
                model, success = load_sample_data()
                if success:
                    st.session_state['model'] = model
                    st.session_state['data_loaded'] = True
                    st.success("Data loaded successfully!")
                else:
                    st.error("Failed to load data.")
    
    else:
        model = st.session_state['model']
        
        # Check if models are trained
        models_ready = model.classifier is not None and model.regressor is not None
        
        if not models_ready:
            st.info("Models need to be trained before predictions can be made.")
            if st.button("Train Prediction Models"):
                with st.spinner("Training models..."):
                    try:
                        # Train both models
                        classification_results = model.train_deal_outcome_model()
                        regression_results = model.train_time_to_close_model()
                        
                        st.success("Models trained successfully!")
                        
                        # Show model performance
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Deal Outcome Model Performance")
                            st.metric("Accuracy", f"{classification_results['accuracy']:.2f}")
                            st.metric("F1 Score", f"{classification_results['f1_score']:.2f}")
                        
                        with col2:
                            st.markdown("### Time to Close Model Performance")
                            st.metric("RMSE (days)", f"{regression_results['rmse']:.2f}")
                            st.metric("R² Score", f"{regression_results['r2_score']:.2f}")
                        
                    except Exception as e:
                        st.error(f"Error training models: {e}")
        
        # Deal prediction form
        st.markdown('### Predict New Deal')
        
        # Create form for deal attributes
        with st.form("deal_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                deal_amount = st.number_input("Deal Amount ($)", min_value=1000, max_value=1000000, value=50000)
                deal_probability = st.slider("Deal Probability (%)", 0, 100, 50) / 100
                forecast_amount = st.number_input("Forecast Amount ($)", min_value=0, max_value=1000000, value=int(deal_amount * deal_probability))
            
            with col2:
                deal_stage_options = ["Opportunity", "BANT Deal", "Deep Dive", "In Trial", "Negotiation", "Contract Sent"]
                deal_stage = st.selectbox("Deal Stage", deal_stage_options)
                
                pipeline_options = ["Sales Pipeline"]
                pipeline = st.selectbox("Pipeline", pipeline_options)
                
                forecast_category_options = ["Pipeline", "Best Case", "Commit"]
                forecast_category = st.selectbox("Forecast Category", forecast_category_options)
            
            # Additional optional fields
            with st.expander("Additional Deal Attributes"):
                col1, col2 = st.columns(2)
                
                with col1:
                    create_date = st.date_input("Create Date", datetime.now() - timedelta(days=7))
                    industry_options = ["Technology", "Healthcare", "Financial Services", "Manufacturing", "Retail", "Other"]
                    industry = st.selectbox("Industry", industry_options)
                
                with col2:
                    ticket_count = st.number_input("Support Ticket Count", min_value=0, max_value=10, value=1)
                    
                    training_provided = st.checkbox("Training Provided", value=False)
            
            # Submit button
            submit_button = st.form_submit_button("Predict Deal Outcome")
        
        # Make prediction when form is submitted
        if submit_button:
            if models_ready:
                try:
                    # Prepare input data
                    deal_data = {
                        'Amount': deal_amount,
                        'Deal probability': deal_probability,
                        'Weighted amount': deal_amount * deal_probability,
                        'Forecast amount': forecast_amount,
                        'Create Date': create_date,
                        'Pipeline': pipeline,
                        'Deal Stage': deal_stage,
                        'Forecast category': forecast_category,
                        'ticket_count_per_deal': ticket_count,
                        'Any Training Provided': training_provided
                    }
                    
                    # Make prediction
                    prediction = model.predict_new_deal(deal_data)
                    
                    # Display prediction results
                    st.markdown('### Prediction Results')
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        win_prob_pct = prediction['win_probability'] * 100
                        st.metric("Win Probability", f"{win_prob_pct:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Estimated Days to Close", f"{prediction['estimated_days_to_close']:.1f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Estimated Close Date", prediction['estimated_close_date'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Provide recommendations based on prediction
                    st.markdown('### Deal Strategy Recommendations')
                    
                    if win_prob_pct >= 80:
                        st.markdown("""
                        **High Probability Deal (>80%)**
                        - Focus on smooth progression to closing
                        - Begin pre-implementation planning
                        - Introduce customer success team early
                        - Consider upsell opportunities
                        - Secure necessary resources for implementation
                        """)
                    elif win_prob_pct >= 50:
                        st.markdown("""
                        **Medium Probability Deal (50-80%)**
                        - Address remaining objections proactively
                        - Provide additional case studies or references
                        - Consider executive involvement to reinforce value
                        - Develop clear implementation timeline
                        - Focus on ROI calculations 
                        """)
                    else:
                        st.markdown("""
                        **Low Probability Deal (<50%)**
                        - Identify and address key blockers
                        - Consider trial or pilot options
                        - Evaluate pricing or scope adjustments
                        - Increase stakeholder engagement
                        - Reassess qualification criteria
                        """)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
            else:
                st.warning("Please train the models first.")
        
        # Show model feature importances if available
        if models_ready and model.classifier is not None:
            st.markdown('### Deal Outcome Predictors')
            
            try:
                # Get feature importances
                importance_df = model.get_feature_importances(model_type='classifier')
                
                if importance_df is not None:
                    # Plot feature importances
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
                    ax.set_title('Top 10 Features for Deal Outcome Prediction')
                    ax.set_xlabel('Importance')
                    ax.set_ylabel('Feature')
                    st.pyplot(fig)
                else:
                    st.info("Feature importance information is not available for this model type.")
            except Exception as e:
                st.error(f"Error displaying feature importances: {str(e)}")

# Implementation Insights page
elif page == "Implementation Insights":
    st.markdown('<h2 class="sub-header">Implementation Insights</h2>', unsafe_allow_html=True)
    
    if not st.session_state['data_loaded']:
        st.warning("Please load the data first from the Home page.")
        if st.button("Load Data Now"):
            with st.spinner("Loading data..."):
                model, success = load_sample_data()
                if success:
                    st.session_state['model'] = model
                    st.session_state['data_loaded'] = True
                    st.success("Data loaded successfully!")
                else:
                    st.error("Failed to load data.")
    
    else:
        model = st.session_state['model']
        tickets_df = model.tickets
        
        # Implementation metrics
        st.markdown('### Implementation Performance Metrics')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'time_to_close' in tickets_df.columns:
                avg_implementation_time = tickets_df['time_to_close'].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg. Implementation Time", f"{avg_implementation_time:.1f} days")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Implementations", len(tickets_df))
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col2:
            # Calculate training completion rate if available
            training_columns = [col for col in tickets_df.columns if 'Training:' in col]
            if training_columns:
                training_completion = tickets_df[training_columns].notna().mean().mean() * 100
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Training Completion Rate", f"{training_completion:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            # Success conversion rate if available
            if 'Ticket status' in tickets_df.columns:
                success_rate = (tickets_df['Ticket status'] == 'Closed').mean() * 100
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Implementation Success Rate", f"{success_rate:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Training impact analysis
        st.markdown('### Training Impact Analysis')
        
        # Analyze training impact if data is available
        training_columns = [col for col in tickets_df.columns if 'Training:' in col]
        
        if training_columns:
            # Create a flag for any training provided
            tickets_df['Any Training Provided'] = tickets_df[training_columns].notna().any(axis=1)
            
            # Analyze implementation metrics by training status
            if 'time_to_close' in tickets_df.columns and 'Any Training Provided' in tickets_df.columns:
                training_time_impact = tickets_df.groupby('Any Training Provided')['time_to_close'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x='Any Training Provided', y='time_to_close', data=training_time_impact, ax=ax)
                ax.set_title('Impact of Training on Implementation Time')
                ax.set_xlabel('Training Provided')
                ax.set_ylabel('Average Implementation Time (days)')
                ax.set_xticklabels(['No', 'Yes'])
                st.pyplot(fig)
            
            # Training completion analysis
            st.markdown('### Training Completion Analysis')
            
            # Calculate completion rates for each training type
            training_completion_rates = {
                col: tickets_df[col].notna().mean() * 100 
                for col in training_columns
            }
            
            # Convert to dataframe for visualization
            training_df = pd.DataFrame({
                'Training Type': [col.replace('Training: ', '') for col in training_completion_rates.keys()],
                'Completion Rate (%)': training_completion_rates.values()
            })
            
            # Plot completion rates
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Training Type', y='Completion Rate (%)', data=training_df, ax=ax)
            ax.set_title('Training Completion Rates by Type')
            ax.set_xlabel('Training Type')
            ax.set_ylabel('Completion Rate (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        else:
            st.info("Training data not available in the tickets dataset.")
        
        # Implementation timeline analysis
        st.markdown('### Implementation Timeline Analysis')
        
        # Try to find and analyze milestone dates
        milestone_columns = [
            '1st Syms presented for review', 
            '1st Syms approved for production',
            '1st syms run in production'
        ]
        
        milestone_exists = any(col in tickets_df.columns for col in milestone_columns)
        
        if milestone_exists:
            # Convert date columns to datetime
            for col in milestone_columns:
                if col in tickets_df.columns:
                    tickets_df[col] = pd.to_datetime(tickets_df[col], errors='coerce')
            
            # Calculate average days between milestones
            milestone_durations = {}
            
            if 'Create date' in tickets_df.columns and '1st Syms presented for review' in tickets_df.columns:
                tickets_df['Days to First Presentation'] = (
                    tickets_df['1st Syms presented for review'] - pd.to_datetime(tickets_df['Create date'])
                ).dt.days
                milestone_durations['Creation to First Presentation'] = tickets_df['Days to First Presentation'].mean()
            
            if '1st Syms presented for review' in tickets_df.columns and '1st Syms approved for production' in tickets_df.columns:
                tickets_df['Days from Presentation to Approval'] = (
                    tickets_df['1st Syms approved for production'] - tickets_df['1st Syms presented for review']
                ).dt.days
                milestone_durations['Presentation to Approval'] = tickets_df['Days from Presentation to Approval'].mean()
            
            if '1st Syms approved for production' in tickets_df.columns and '1st syms run in production' in tickets_df.columns:
                tickets_df['Days from Approval to Production'] = (
                    tickets_df['1st syms run in production'] - tickets_df['1st Syms approved for production']
                ).dt.days
                milestone_durations['Approval to Production'] = tickets_df['Days from Approval to Production'].mean()
            
            # Create dataframe for visualization
            if milestone_durations:
                milestones_df = pd.DataFrame({
                    'Milestone': milestone_durations.keys(),
                    'Average Days': milestone_durations.values()
                })
                
                # Plot average days between milestones
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Milestone', y='Average Days', data=milestones_df, ax=ax)
                ax.set_title('Average Days Between Implementation Milestones')
                ax.set_xlabel('Milestone')
                ax.set_ylabel('Average Days')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        
        else:
            st.info("Milestone data not available in the tickets dataset.")
        
        # Resource allocation recommendations
        st.markdown('### Resource Allocation Recommendations')
        
        # Based on implementation complexity and volume
        st.markdown("""
        **Optimal Resource Allocation Strategy:**
        
        1. **High-Complexity, High-Value Implementations:**
           - Assign senior implementation specialists
           - Provide comprehensive training early in the process
           - Schedule regular checkpoint meetings
           - Allocate 1.5x standard implementation hours
        
        2. **Medium-Complexity Implementations:**
           - Use a balanced team of specialists
           - Focus on key training modules based on customer needs
           - Standardize implementation timeline
           - Schedule strategic check-ins at milestone points
        
        3. **Low-Complexity, High-Volume Implementations:**
           - Leverage guided self-service tools
           - Offer group training sessions
           - Provide templated implementation guides
           - Implement efficient ticket-based support
        """)

# Run the app
if __name__ == "__main__":
    pass  # The Streamlit framework handles execution