import pandas as pd
import streamlit as st

# Function to load and process data
def load_and_process_data(file):
    # Read the Excel file
    df = pd.read_excel(file)
    
    # Normalize data into lowercase
    df['position'] = df['position'].str.lower()
    
    # Calculate the duration spent inside and outside for each day
    duration_df = df.groupby(['date', 'position']).size().reset_index(name='duration')
    
    # Pivot the duration_df to have separate columns for inside and outside
    duration_pivot_df = duration_df.pivot(index='date', columns='position', values='duration').reset_index()
    
    # Fill NaN values with 0
    duration_pivot_df = duration_pivot_df.fillna(0)
    
    # Calculate the number of picking and placing activities for each day
    activity_df = df.groupby(['date', 'activity']).size().reset_index(name='count')
    
    # Pivot the activity_df to have separate columns for picking and placing
    activity_pivot_df = activity_df.pivot(index='date', columns='activity', values='count').reset_index()
    
    # Fill NaN values with 0
    activity_pivot_df = activity_pivot_df.fillna(0)
    
    # Merge the two DataFrames on the 'date' column
    final_df = pd.merge(duration_pivot_df, activity_pivot_df, on='date')
    
    return duration_pivot_df, activity_pivot_df, final_df

# Streamlit app
st.title('Data Processing and Analysis')

# File upload
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    duration_pivot_df, activity_pivot_df, final_df = load_and_process_data(uploaded_file)
    
    st.subheader('Datewise Total Duration for Each Inside and Outside:')
    st.dataframe(duration_pivot_df)
    
    st.subheader('Datewise Number of Picking and Placing Activity Done:')
    st.dataframe(activity_pivot_df)
    
    st.subheader('Merged Data:')
    st.dataframe(final_df)
