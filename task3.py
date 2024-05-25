import pandas as pd


# Convert to DataFrame
df = pd.read_excel('rawdata.xlsx')

#normalizing data into lowecase
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

# Display the results
print("Datewise Total Duration for Each Inside and Outside:")
print(duration_pivot_df)
print("\nDatewise Number of Picking and Placing Activity Done:")
print(activity_pivot_df)



# Assuming you already have duration_pivot_df and activity_pivot_df DataFrames

# Merge the two DataFrames on the 'date' column
final_df = pd.merge(duration_pivot_df, activity_pivot_df, on='date')

# Print the final DataFrame
print(final_df)
