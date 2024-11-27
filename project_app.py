import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Set the page configuration
st.set_page_config(
    page_title="Job Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title
st.title("📊 Job Data Dashboard")
st.write("Analyze and visualize job data to uncover insights.")

# File uploader
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if (True):
    # Load the dataset
    data = pd.read_excel("https://github.com/AnkitKolhe149/job-hunt-analysis/raw/0ffa08e92b4c4da6add68b47a3136081eb533868/Updated_JOBLIST1.xlsx")

    # Clean the data
    # Drop unnecessary columns (e.g., 'Unnamed: 4') if they exist
    data = data.drop(columns=['Unnamed: 4'], errors='ignore')

    # Drop rows with missing values in the 'Salary' column
    data = data.dropna(subset=['Salary'])

    # Salary extraction function
    def extract_salary(salary_str):
        salary_str = str(salary_str)
        salary_range = []

        # Split the salary range (e.g., "₹3L - ₹4L" or "₹40T - ₹50T")
        for s in salary_str.split('-'):
            s = s.strip().replace('₹', '').replace(',', '')  # Remove currency symbols and commas

            # Convert 'L' to lakh (1L = 100,000) and 'T' to thousand (1T = 1,000)
            if 'L' in s:
                salary_range.append(float(s.replace('L', '')) * 100000)
            elif 'T' in s:
                salary_range.append(float(s.replace('T', '')) * 1000)
            else:
                salary_range.append(float(s))  # Handle plain numeric values if any

        return np.mean(salary_range) if salary_range else None  # Use the average of the range

    # Apply the function to the Salary column
    data['Salary'] = data['Salary'].apply(extract_salary)

    # Apply log transformation to the Salary column
    data['Salary'] = np.log1p(data['Salary'])

    # Display dataset summary
    st.write("## Dataset Overview")
    st.dataframe(data.head())
    st.write(f"*Total Records:* {len(data)}")
    st.write(f"*Unique Jobs:* {data['Job_Title'].nunique()}")

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")
    job_filter = st.sidebar.multiselect(
        "Filter by Job Title", 
        options=data["Job_Title"].unique(), 
        default=data["Job_Title"].unique()
    )
    location_filter = st.sidebar.multiselect(
        "Filter by Location", 
        options=data["Location"].unique(), 
        default=data["Location"].unique()
    )

    # Apply filters
    filtered_data = data[ 
        (data["Job_Title"].isin(job_filter)) & 
        (data["Location"].isin(location_filter))
    ]
    
    # Display filtered dataset
    st.write("## Filtered Data")
    st.dataframe(filtered_data)
    st.write(f"*Filtered Records:* {len(filtered_data)}")
    st.write(f"*Unique Jobs in Filtered Data:* {filtered_data['Job_Title'].nunique()}")

    # Define features and target variable for prediction
    X = filtered_data[['Job_Title', 'Company', 'Location']]  # Assuming these are the features after encoding
    y = filtered_data['Salary']

    # Feature encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_features = encoder.fit_transform(X)


    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(encoded_features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

    # Set up the Ridge Regression model
    ridge = Ridge()

    # Hyperparameter grid search for 'alpha'
    param_grid = {'alpha': np.logspace(-6, 6, 13)}  # Search for a wide range of alpha values
    grid_search = GridSearchCV(ridge, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best alpha found through grid search
    best_alpha = grid_search.best_params_['alpha']
    st.write(f"Best alpha: {best_alpha}")

    # Train the Ridge model with the best alpha
    ridge_best = Ridge(alpha=best_alpha)
    ridge_best.fit(X_train, y_train)

    # Predict on the test set
    y_pred = ridge_best.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"*Mean Squared Error (MSE):* {mse:.3f}")
    st.write(f"*R-squared (R²):* {r2:.3f}")

# Add Prediction Section
    st.write("## 🎯 Salary Prediction")
    with st.form("prediction_form"):
        st.write("### Enter Job Details")
        input_role = st.selectbox("Select Job Role", options=data['Job_Title'].unique(), help="Choose the job title you are interested in.")
        input_location = st.selectbox("Select Location", options=data['Location'].unique(), help="Choose the location for the job.")
        input_company = st.text_input("Enter Company Name", help="Optionally, enter the company name for better predictions.")
        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        input_data = pd.DataFrame({'Job_Title': [input_role], 'Company': [input_company], 'Location': [input_location]})
        encoded_input = encoder.transform(input_data)
        scaled_input = scaler.transform(encoded_input)
        predicted_salary = ridge_best.predict(scaled_input)
        predicted_salary_exp = np.expm1(predicted_salary)

        st.write("### Predicted Salary")
        st.metric("Predicted Salary (Approx.)", f"₹{predicted_salary_exp[0]:,.2f}")


    # Visualization Options
    st.sidebar.header("📈 Visualization Options")
    viz_option = st.sidebar.selectbox(
        "Choose Visualization Type",
        ("Salary Analysis", "Notice Period Trends", "Job Count Analysis", 
         "Correlation Analysis", "House Allowance Analysis")
    )

    # Plot customization
    st.sidebar.header("🎨 Plot Customization")
    color_palette = st.sidebar.selectbox(
        "Select a Color Palette", 
        ["husl", "coolwarm", "viridis", "Set2", "Set3", "Blues", "Greens"]
    )
    palette = sns.color_palette(color_palette)

    # Group By Option
    group_by = st.sidebar.selectbox(
        "Group By", 
        options=["Job_Title", "Location"], 
        index=0
    )

    # Salary Analysis
    if viz_option == "Salary Analysis":
        st.write("## Salary Analysis")
        salary_metric = st.selectbox("Choose Salary Metric", ["Minimum_Salary", "Maximum_Salary"])

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=filtered_data, 
            x=group_by, 
            y=salary_metric, 
            ci=None, 
            palette=palette, 
            ax=ax
        )
        plt.xticks(rotation=45)
        plt.title(f"{salary_metric} by {group_by}", fontsize=16)
        plt.xlabel(group_by, fontsize=14)
        plt.ylabel(salary_metric, fontsize=14)
        
        # Adding data labels to the bars
        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=12, 
                color='black', 
                xytext=(0, 10), 
                textcoords='offset points'
            )
        
        st.pyplot(fig)

        avg_salary = filtered_data[salary_metric].mean()
        st.metric(label=f"Average {salary_metric}", value=f"{avg_salary:.2f} L")

    # Notice Period Trends
    elif viz_option == "Notice Period Trends":
        st.write("## Notice Period Trends")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=filtered_data, 
            x=group_by, 
            y="Minimum_Notice_Period", 
            palette=palette, 
            ax=ax
        )
        plt.xticks(rotation=45)
        plt.title(f"Notice Period Distribution by {group_by}", fontsize=16)
        plt.xlabel(group_by, fontsize=14)
        plt.ylabel("Minimum Notice Period (months)", fontsize=14)
        
        # Adjusting tick marks
        ax.set_yticks(range(0, int(filtered_data['Minimum_Notice_Period'].max()) + 1, 1))

        st.pyplot(fig)

    # Job Count Analysis
    elif viz_option == "Job Count Analysis":
        st.write("## Job Count Analysis")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.countplot(
            data=filtered_data, 
            x=group_by, 
            order=filtered_data[group_by].value_counts().index, 
            palette=palette, 
            ax=ax
        )
        plt.xticks(rotation=45)
        plt.title(f"Job Count by {group_by}", fontsize=16)
        plt.xlabel(group_by, fontsize=14)
        plt.ylabel("Job Count", fontsize=14)
        
        # Adding job count labels on top of bars
        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=12, 
                color='black', 
                xytext=(0, 10), 
                textcoords='offset points'
            )
        
        st.pyplot(fig)

    # Correlation Analysis
    elif viz_option == "Correlation Analysis":
        st.write("## Correlation Analysis")
        correlation_matrix = filtered_data[["Minimum_Salary", "Maximum_Salary", "Minimum_Notice_Period"]].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax, annot_kws={"size": 12}, fmt=".2f")
        plt.title("Correlation Matrix", fontsize=16)
        st.pyplot(fig)

    # House Allowance Analysis
    elif viz_option == "House Allowance Analysis":
        st.write("## House Allowance by Job Title")
        if "House_Allowance" in filtered_data.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=filtered_data,
                x=group_by,
                y="House_Allowance",
                ci=None,
                palette=palette,
                ax=ax
            )
            plt.xticks(rotation=45)
            plt.title("House Allowance by Job Title", fontsize=16)
            plt.xlabel(group_by, fontsize=14)
            plt.ylabel("House Allowance (L)", fontsize=14)
            
            # Adding data labels
            for p in ax.patches:
                ax.annotate(
                    f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', 
                    va='center', 
                    fontsize=12, 
                    color='black', 
                    xytext=(0, 10), 
                    textcoords='offset points'
                )
            
            st.pyplot(fig)

            avg_allowance = filtered_data.groupby(group_by)["House_Allowance"].mean()
            st.write(f"### Average House Allowance by {group_by}")
            st.dataframe(avg_allowance)
        else:
            st.write("The column 'House_Allowance' is not available in the dataset.")

    # Summary in Sidebar
    st.sidebar.write("---")
    st.sidebar.subheader("📋 Dataset Summary")
    st.sidebar.write(f"*Total Jobs:* {len(filtered_data)}")
    st.sidebar.write(f"*Unique Jobs:* {filtered_data['Job_Title'].nunique()}")
    st.sidebar.write(f"*Average Minimum Salary:* {filtered_data['Minimum_Salary'].mean():.2f} L")
    st.sidebar.write(f"*Average Maximum Salary:* {filtered_data['Maximum_Salary'].mean():.2f} L")

else:
    st.write("👆 Please upload an Excel file to get started.")
