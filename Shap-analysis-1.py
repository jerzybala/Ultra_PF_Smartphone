# %%
import pandas as pd
import numpy as np
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import math


# Read the data into a pandas DataFrame

path = '/Users/jerzybala/Desktop/Simulation Experiments Aug-Sep 2023/PROCESSED_FOOD_Dec2023.csv'

path2 = '/Volumes/Desktop/Simulation Experiments Aug-Sep 2023/PROCESSED_FOOD_Dec2023.csv'

df_org = pd.read_csv(path)

print(df_org.shape)
#df_org.columns.tolist()

# # Calculate the original distribution
# original_distribution = df_org['Processed food in diet'].value_counts()
# print(original_distribution,"\n")



region_mapping = {
    'Anglosphere': ['United States', 'Canada', 'United Kingdom', 'Ireland', 'New Zealand', 'Australia'],
    'Latin America': ['Argentina', 'Chile', 'Colombia', 'Ecuador', 'Guatemala', 'Mexico', 'Peru', 'Puerto Rico', 'Venezuela', 'Brazil', 'Bolivia', 'Paraguay', 'Uruguay'],
    'Middle East': ['Iraq', 'Saudi Arabia', 'United Arab Emirates', 'Yemen', 'Iran', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Qatar', 'Syria', 'Bahrain'],
    'French & Spanish Speaking Mainland Europe': ['Spain', 'France', 'Belgium', 'Switzerland', 'Portugal', 'Italy', 'Greece', 'Germany', 'Austria', 'Netherlands'],
    'West Africa': ['Cameroon', 'Cote d’Ivoire', 'Democratic Republic of the Congo', 'Nigeria', 'Ghana', 'Senegal', 'Mali', 'Niger', 'Guinea'],
    'North Africa': ['Algeria', 'Egypt', 'Morocco', 'Tunisia', 'Libya', 'Sudan'],
    'East & Southeast Asia': ['China', 'Japan', 'South Korea', 'North Korea', 'Vietnam', 'Thailand', 'Malaysia', 'Indonesia', 'Philippines', 'Singapore'],
    'South Asia': ['India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Bhutan'],
    'Eastern Europe & Central Asia': ['Russia', 'Ukraine', 'Belarus', 'Moldova', 'Armenia', 'Azerbaijan', 'Georgia', 'Kazakhstan', 'Uzbekistan', 'Turkmenistan', 'Kyrgyzstan', 'Tajikistan'],
    'Scandinavia & Baltic': ['Sweden', 'Norway', 'Finland', 'Denmark', 'Iceland', 'Estonia', 'Latvia', 'Lithuania'],
    'Oceania & Pacific Islands': ['Fiji', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga'],
    'Sub-Saharan Africa': ['Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Burundi', 'South Africa', 'Zimbabwe', 'Zambia', 'Botswana', 'Namibia', 'Mozambique', 'Madagascar'],
    'Central America & Caribbean': ['Cuba', 'Dominican Republic', 'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Barbados', 'Saint Lucia', 'Grenada', 'Belize', 'Costa Rica', 'El Salvador', 'Honduras', 'Nicaragua', 'Panama']
    # ... You can continue to add more regions and countries as needed.
}


# Invert the region mapping
country_to_region = {country: region for region, countries in region_mapping.items() for country in countries}

# Map each country to its region
df_org['Region'] = df_org['Country'].map(country_to_region)

# Now 'data' has an additional column 'Region' which indicates the region of each country



set1 = [
 'Overall MHQ',
 'MHQ_Sign',
 
 'Age',
 'Biological Sex',
#  'Different gender from biological sex',
#  'Gender Identity',
#  'Ethnicity',
 'Country',
#  'State',
#  'Place of living',
#  'City',
 'Education',
 'Employment',#  Yes
 'Region',
#  'Profession/Employment Sector',
 
 'Frequency of getting a good nights sleep',
 'Frequency of doing exercise',
 'Processed food in diet',
 'Frequency of Socializing',
 
#  'Number of Children',
#  'Household Size',
#  'Number of Siblings when growing up',
#  'Number of Close Friends',
 
#  'Description of Household Growing up',
#  'Relationship with Adult Family',
#  'Spirituality connection',
#  'Feelings of love towards others',
#  'Particular religion identity',
#  'Religion practice',

# smartphone with age<24
 
#  'Smartphone allowed in school',
#  'Smartphone use in lessons',
#  'Age of first social media account',
#  'Frequency of social media posting',
#  'Tablet ownership',
#  'Smartphone ownership',
#  'Friends/classmates smarphone ownership',
#  'Age of smartphone usage during school hours',
#  'Smartphone usage during class hours',
#  'Smartphone usage during break',
#  'Age of laptop usage required by school',
#  'Laptop usage for non-learning activities',

 'Sudden or premature death of a loved one',
 'Divorce/separation  or family breakup',
 'Extreme poverty leading to homelessness and/or hunger.',
 'Forced family control over major life decisions (e.g. marriage)',
 'Prolonged sexual abuse| or severe sexual assault.',
 'Displacement from your home due to political| environmental or economic reasons',
 'Loss of your job or livelihood leading to an inability to make ends meet.',
 'Cyberbullying or online abuse',
 'Threatening| coercive or controlling behavior by another person',
 'Caring for a child or partner with a major chronic disability or illness',
 'I did not experience any of the above',
 'Involvement or close witness to a war',
 'Life threatening or debilitating injury or illness.',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster',
 'None of the above AT',

#  'Type II Diabetes',
#  'Fibromyalgia',
#  'Liver disease/Cirrhosis',
#  'Hypertension',
#  'Irritable Bowel Syndrome',
#  'Heart disease',
#  "Inflammatory Bowel Disease / Crohn's disease",
#  'Arthritis',
#  'Psoriasis',
#  'Asthma',
#  'Migraines',
#  'Traumatic Brain Injury',
#  'Osteoporosis',
#  'Sleep apnea',
#  'Hipotiroidismo',
#  'Neuropathy',
#  'Cancer',
#  'Chronic Obstructive Pulmonary Disease (COPD)',
#  'HIV /AIDS',
#  'Kidney Disease',
#  'Rheumatoid',
#  'Polycystic ovaries',
#  'Prefer not to say',
#  'Narcolepsy',
#  'Chronic fatigue syndrome',
#  'Back problem',
#  'Epilepsy',
#  'Multiple sclerosis',
#  'Escoliosis',
#  'Stroke',
#  'Herpes',
#  'Type 1 Diabetes',

 'Sudden or premature death of a parent or sibling',
 'Prolonged emotional or psychological abuse or neglect from parent/caregiver',
 'Prolonged physical abuse| or severe physical assault CT',
 'Physical violence in the home between family members',
 'Prolonged or sustained bullying in person from peers',
 'Parental Divorce or family breakup',
 'Lived with a parent/caregiver who was an alcoholic or who regularly used street drugs',
 'Threatening| coercive or controlling behavior by another person CT',
 'I did not experience any of the above during my childhood',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster CT',
 'Displacement from your home due to political| environmental or economic reasons CT',
 'Life threatening or debilitating injury or illness CT',
 'Forced family control over major life decisions CT',
 'None of the above CT',

 'Tobacco products',
 'Alcoholic beverages',
 'Cannabis',
 'Vaping products',
 'Sedatives or Sleeping Pills',
 'Amphetamine type stimulants (e.g. speed| diet pills| ecstasy| etc.)',
 'Opioids',
 'Melatonina',
 'Cigarrillo',
 'None of the above']


df_org2=df_org[set1].copy()




df_org2.loc[:, 'Processed food in diet'] = df_org2['Processed food in diet'].replace({
    'Rarely/never': 'Rarely/Never',
    'A few times in a day': 'Several times a day',
    'Several days a week': 'A few times a week',
    'Many times in a day': 'Several times a day',
    'At least once a day': 'Several times a day'
})


print(df_org2.shape, "\n")



# %% [markdown]
# ## Transitions
# 

# %%
# # Segmenting the data based on 'Processed food in diet'
# rarely_never = df[df['Processed food in diet'] == 'Rarely/Never']
# once_a_day = df[df['Processed food in diet'] == 'Once a day']

# # Calculate frequency counts for each feature in 'Rarely/Never' segment
# rn_counts = {
#     'Frequency of getting a good nights sleep': rarely_never['Frequency of getting a good nights sleep'].value_counts(),
#     'Frequency of doing exercise': rarely_never['Frequency of doing exercise'].value_counts(),
#     'Frequency of Socializing': rarely_never['Frequency of Socializing'].value_counts()
# }


# %%
# #Function to assess similarity using chi-squared test
# from scipy.stats import chisquare


# def is_similar(row, rn_counts):
#     p_values = []
#     for feature, rn_count in rn_counts.items():
#         # Create observed frequency counts
#         observed = once_a_day[feature].value_counts()

#         # Reindex the expected frequency counts to match observed, fill missing with 0
#         expected = rn_count.reindex(observed.index, fill_value=0)

#         # Adjust expected frequencies to have the same sum as observed
#         expected = expected * (observed.sum() / expected.sum())

#         # Chi-squared test
#         chi2, p = chisquare(f_obs=observed, f_exp=expected)
#         p_values.append(p)

#     # Check if all p-values are above a significance level (e.g., 0.05)
#     return all(p_val > 0.01 for p_val in p_values)

# %%
# # Identify similar rows
# similar_rows_indices = once_a_day[once_a_day.apply(lambda row: is_similar(row, rn_counts), axis=1)].index


# # Change 'Processed food in diet' for similar rows in a new dataframe
# df1 = df.copy()
# df1.loc[similar_rows_indices, 'Processed food in diet'] = 'Rarely/Never'

# %%
# df2['Processed food in diet'].value_counts()


# %%


# %%



# %%
# df1 = df.copy()
# df1.loc[rows_to_change, 'Processed food in diet'] = 'Rarely/Never'

# # Optionally, remove the 'distance' column from df1
# df1.drop(columns=['distance'], inplace=True)

# # Check the modified distribution
# print(df1['Processed food in diet'].value_counts())

# %%


# %%

# # Segmenting the data based on 'Processed food in diet'
# rarely_never = df[df['Processed food in diet'] == 'Rarely/Never']
# once_a_day = df[df['Processed food in diet'] == 'Once a day']

# # Calculate mode (most common value) for each feature in 'Rarely/Never' segment
# rn_modes = {
#     'Frequency of getting a good nights sleep': rarely_never['Frequency of getting a good nights sleep'].mode()[0],
#     'Frequency of doing exercise': rarely_never['Frequency of doing exercise'].mode()[0],
#     'Frequency of Socializing': rarely_never['Frequency of Socializing'].mode()[0]
# }

# # Function to check if row matches the mode values
# def matches_mode(row, rn_modes):
#     return all(row[feature] == rn_mode for feature, rn_mode in rn_modes.items())

# # Identify rows in 'Once a day' that match the mode of 'Rarely/Never'
# matching_rows_indices = once_a_day[once_a_day.apply(lambda row: matches_mode(row, rn_modes), axis=1)].index



df3=df_org2.copy()


rarely_never = df3[df3['Processed food in diet'] == 'Rarely/Never']
once_a_day = df3[df3['Processed food in diet'] == 'Once a day']

# Calculate mode (most common value) for each feature in 'Rarely/Never' segment
rn_modes = {
    'Frequency of getting a good nights sleep': rarely_never['Frequency of getting a good nights sleep'].mode()[0],
    'Frequency of doing exercise': rarely_never['Frequency of doing exercise'].mode()[0],
    'Frequency of Socializing': rarely_never['Frequency of Socializing'].mode()[0]
}

# Function to check if a row matches the mode values in at least two out of the three specified features
def matches_mode_specific_features(row, rn_modes):
    match_count = sum(row[feature] == rn_mode for feature, rn_mode in rn_modes.items())
    return match_count >= 2  # At least two matches out of the three specified features

# Identify rows in 'Once a day' that match the mode of 'Rarely/Never' in the specified features
matching_rows_indices = once_a_day[once_a_day.apply(lambda row: matches_mode_specific_features(row, rn_modes), axis=1)].index

# Change 'Processed food in diet' for these rows in a new dataframe

df3.loc[matching_rows_indices, 'Processed food in diet'] = 'Rarely/Never'

# Check the modified distribution
print(df3['Processed food in diet'].value_counts())



# %%
print(df_org2['Processed food in diet'].value_counts())



# %%


# %%


# %%
# org2.copy()   original
# df3 after redistribution

# %% [markdown]
# # Scenario processing

# %% [markdown]
# ## Age

# %%
df=df_org2.copy()
#df = df3

#Age filter
ages = ['18-24', '21-24', '18', '19', '20']
df = df[df['Age'].isin(ages)]

# Select only the columns with object type (commonly used for categorical features)
categorical_features = df.select_dtypes(include=['object']).columns
# Apply pd.get_dummies to the categorical features with a prefix
encoded_features = pd.get_dummies(df[categorical_features])
# Concatenate the encoded features with the original DataFrame (excluding the original categorical features)

df = pd.concat([df.drop(columns=categorical_features), encoded_features], axis=1)

df.columns.tolist()

to_model_a = [
 'Overall MHQ',
 'MHQ_Sign',
 'Sudden or premature death of a loved one',
 'Divorce/separation  or family breakup',
 'Extreme poverty leading to homelessness and/or hunger.',
 'Forced family control over major life decisions (e.g. marriage)',
 'Prolonged sexual abuse| or severe sexual assault.',
 'Displacement from your home due to political| environmental or economic reasons',
 'Loss of your job or livelihood leading to an inability to make ends meet.',
 'Cyberbullying or online abuse',
 'Threatening| coercive or controlling behavior by another person',
 'Caring for a child or partner with a major chronic disability or illness',
 'I did not experience any of the above',
 'Involvement or close witness to a war',
 'Life threatening or debilitating injury or illness.',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster',
 'None of the above AT',
 'Sudden or premature death of a parent or sibling',
 'Prolonged emotional or psychological abuse or neglect from parent/caregiver',
 'Prolonged physical abuse| or severe physical assault CT',
 'Physical violence in the home between family members',
 'Prolonged or sustained bullying in person from peers',
 'Parental Divorce or family breakup',
 'Lived with a parent/caregiver who was an alcoholic or who regularly used street drugs',
 'Threatening| coercive or controlling behavior by another person CT',
 'I did not experience any of the above during my childhood',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster CT',
 'Displacement from your home due to political| environmental or economic reasons CT',
 'Life threatening or debilitating injury or illness CT',
 'Forced family control over major life decisions CT',
 'None of the above CT',
 'Tobacco products',
 'Alcoholic beverages',
 'Cannabis',
 'Vaping products',
 'Sedatives or Sleeping Pills',
 'Amphetamine type stimulants (e.g. speed| diet pills| ecstasy| etc.)',
 'Opioids',
 'Melatonina',
 'Cigarrillo',
 
 'Biological Sex_Female',
 'Biological Sex_Male',
 'Biological Sex_Other/Intersex',
 'Biological Sex_Prefer not to say',
 
 'Frequency of getting a good nights sleep_All of the time',
 'Frequency of getting a good nights sleep_Hardly ever',
 'Frequency of getting a good nights sleep_Most days',
 'Frequency of getting a good nights sleep_Most of the time',
 'Frequency of getting a good nights sleep_Some of the time',
 
 'Frequency of doing exercise_Every day',
 'Frequency of doing exercise_Few days a week',
 'Frequency of doing exercise_Less than once a week',
 'Frequency of doing exercise_Once a week',
 'Frequency of doing exercise_Rarely/Never',
 'Frequency of doing exercise_Several days a week',
 #'Frequency of doing exercise_Some days of the week',
 'Processed food in diet_A few times a month',
 'Processed food in diet_A few times a week',
 'Processed food in diet_Once a day',
 'Processed food in diet_Rarely/Never',
 'Processed food in diet_Several times a day',
 'Frequency of Socializing_1-3 times a month',
 'Frequency of Socializing_Once a week',
 'Frequency of Socializing_Rarely/Never',
 'Frequency of Socializing_Several days a week']

df = df[to_model_a].copy()
print(len(df))




# %% [markdown]
# # Countries

# %%

print(df3['Processed food in diet'].value_counts())
print()
print(df_org2['Processed food in diet'].value_counts())



# %%


# %%
#df_c=df_org2.copy()
df_c = df3

num_empty_spaces = df_c['Processed food in diet'].isna().sum()
print("num_empty_spaces;",num_empty_spaces)


# Check the modified distribution
print(df_c['Processed food in diet'].value_counts())

# Country filter
countries = ['United States','United Kingdom','Australia']
df_c = df_c[df_c['Country'].isin(countries)]
#print(df_c['Country'].value_counts())

df_c = df_c.drop(columns='Country', axis=1)
#print(df_c['Country'].value_counts())


# Check the modified distribution
print(df_c['Processed food in diet'].value_counts())

#print(df['Country'].value_counts())

# Select only the columns with object type (commonly used for categorical features)
categorical_features = df_c.select_dtypes(include=['object']).columns
# Apply pd.get_dummies to the categorical features with a prefix
encoded_features = pd.get_dummies(df_c[categorical_features])
# Concatenate the encoded features with the original DataFrame (excluding the original categorical features)

df_c = pd.concat([df_c.drop(columns=categorical_features), encoded_features], axis=1)

to_model_c = [
    
 'Overall MHQ',
 'MHQ_Sign',
 'Sudden or premature death of a loved one',
 'Divorce/separation  or family breakup',
 'Extreme poverty leading to homelessness and/or hunger.',
 'Forced family control over major life decisions (e.g. marriage)',
 'Prolonged sexual abuse| or severe sexual assault.',
 'Displacement from your home due to political| environmental or economic reasons',
 'Loss of your job or livelihood leading to an inability to make ends meet.',
 'Cyberbullying or online abuse',
 'Threatening| coercive or controlling behavior by another person',
 'Caring for a child or partner with a major chronic disability or illness',
 'I did not experience any of the above',
 'Involvement or close witness to a war',
 'Life threatening or debilitating injury or illness.',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster',
 'None of the above AT',
 'Sudden or premature death of a parent or sibling',
 'Prolonged emotional or psychological abuse or neglect from parent/caregiver',
 'Prolonged physical abuse| or severe physical assault CT',
 'Physical violence in the home between family members',
 'Prolonged or sustained bullying in person from peers',
 'Parental Divorce or family breakup',
 'Lived with a parent/caregiver who was an alcoholic or who regularly used street drugs',
 'Threatening| coercive or controlling behavior by another person CT',
 'I did not experience any of the above during my childhood',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster CT',
 'Displacement from your home due to political| environmental or economic reasons CT',
 'Life threatening or debilitating injury or illness CT',
 'Forced family control over major life decisions CT',
 'None of the above CT',
 'Tobacco products',
 'Alcoholic beverages',
 'Cannabis',
 'Vaping products',
 'Sedatives or Sleeping Pills',
 'Amphetamine type stimulants (e.g. speed| diet pills| ecstasy| etc.)',
 'Opioids',
 'Melatonina',
 'Cigarrillo',
 
 'Age_18',
 'Age_18-24',
 'Age_19',
 'Age_20',
 'Age_21-24',
 'Age_25-34',
 'Age_35-44',
 'Age_45-54',
 'Age_55-64',
 'Age_65-74',
 'Age_75-84',
 'Age_85+',
 'Biological Sex_Female',
 'Biological Sex_Male',
 'Biological Sex_Other/Intersex',
 'Biological Sex_Prefer not to say',

#  'Country_Australia',
#  'Country_United Kingdom',
#  'Country_United States',
 
 
 'Frequency of getting a good nights sleep_All of the time',
 'Frequency of getting a good nights sleep_Hardly ever',
 'Frequency of getting a good nights sleep_Most days',
 'Frequency of getting a good nights sleep_Most of the time',
 'Frequency of getting a good nights sleep_Some of the time',
 'Frequency of doing exercise_Every day',
 'Frequency of doing exercise_Few days a week',
 'Frequency of doing exercise_Less than once a week',
 'Frequency of doing exercise_Once a week',
 'Frequency of doing exercise_Rarely/Never',
 #'Frequency of doing exercise_Several days a week',
 #'Frequency of doing exercise_Some days of the week',
 'Processed food in diet_A few times a month',
 'Processed food in diet_A few times a week',
 'Processed food in diet_Once a day',
 'Processed food in diet_Rarely/Never',
 'Processed food in diet_Several times a day',
 'Frequency of Socializing_1-3 times a month',
 'Frequency of Socializing_Once a week',
 'Frequency of Socializing_Rarely/Never',
 'Frequency of Socializing_Several days a week']


df_c=df_c[to_model_c].copy()
print(len(df_c))



# %%
df=df_org2.copy()

#Age filter
ages = ['18-24', '21-24', '18', '19', '20']
df = df[df['Age'].isin(ages)]

# Country filter
countries = ['United States','United Kingdom','Australia']
df = df[df['Country'].isin(countries)]
#print(df_c['Country'].value_counts())

print(df['Country'].value_counts())


print(len(df))

# Check the modified distribution
print(df['Processed food in diet'].value_counts())


# Select only the columns with object type (commonly used for categorical features)
categorical_features = df.select_dtypes(include=['object']).columns
# Apply pd.get_dummies to the categorical features with a prefix
encoded_features = pd.get_dummies(df[categorical_features])
# Concatenate the encoded features with the original DataFrame (excluding the original categorical features)

df = pd.concat([df.drop(columns=categorical_features), encoded_features], axis=1)


to_model_ac = [

 'Overall MHQ',
 'MHQ_Sign',
 'Sudden or premature death of a loved one',
 'Divorce/separation  or family breakup',
 'Extreme poverty leading to homelessness and/or hunger.',
 'Forced family control over major life decisions (e.g. marriage)',
 'Prolonged sexual abuse| or severe sexual assault.',
 'Displacement from your home due to political| environmental or economic reasons',
 'Loss of your job or livelihood leading to an inability to make ends meet.',
 'Cyberbullying or online abuse',
 'Threatening| coercive or controlling behavior by another person',
 'Caring for a child or partner with a major chronic disability or illness',
 'I did not experience any of the above',
 'Involvement or close witness to a war',
 'Life threatening or debilitating injury or illness.',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster',
 'None of the above AT',
 'Sudden or premature death of a parent or sibling',
 'Prolonged emotional or psychological abuse or neglect from parent/caregiver',
 'Prolonged physical abuse| or severe physical assault CT',
 'Physical violence in the home between family members',
 'Prolonged or sustained bullying in person from peers',
 'Parental Divorce or family breakup',
 'Lived with a parent/caregiver who was an alcoholic or who regularly used street drugs',
 'Threatening| coercive or controlling behavior by another person CT',
 'I did not experience any of the above during my childhood',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster CT',
 'Displacement from your home due to political| environmental or economic reasons CT',
 'Life threatening or debilitating injury or illness CT',
 'Forced family control over major life decisions CT',
 'None of the above CT',
 'Tobacco products',
 'Alcoholic beverages',
 'Cannabis',
 'Vaping products',
 'Sedatives or Sleeping Pills',
 'Amphetamine type stimulants (e.g. speed| diet pills| ecstasy| etc.)',
 'Opioids',
 'Melatonina',
 'Cigarrillo',
 
 
 
 'Biological Sex_Female',
 'Biological Sex_Male',
 'Biological Sex_Other/Intersex',
 'Biological Sex_Prefer not to say',
 
 
 
 'Frequency of getting a good nights sleep_All of the time',
 'Frequency of getting a good nights sleep_Hardly ever',
 'Frequency of getting a good nights sleep_Most days',
 'Frequency of getting a good nights sleep_Most of the time',
 'Frequency of getting a good nights sleep_Some of the time',
 'Frequency of doing exercise_Every day',
 'Frequency of doing exercise_Few days a week',
 'Frequency of doing exercise_Less than once a week',
 'Frequency of doing exercise_Once a week',
 'Frequency of doing exercise_Rarely/Never',
 'Processed food in diet_A few times a month',
 'Processed food in diet_A few times a week',
 'Processed food in diet_Once a day',
 'Processed food in diet_Rarely/Never',
 'Processed food in diet_Several times a day',
 'Frequency of Socializing_1-3 times a month',
 'Frequency of Socializing_Once a week',
 'Frequency of Socializing_Rarely/Never',
 'Frequency of Socializing_Several days a week']


df=df[to_model_ac].copy()
print(len(df))


# %%
#df.columns.tolist()

# %%
df=df_org2.copy()

# Select only the columns with object type (commonly used for categorical features)
categorical_features = df.select_dtypes(include=['object']).columns
# Apply pd.get_dummies to the categorical features with a prefix
encoded_features = pd.get_dummies(df[categorical_features])
# Concatenate the encoded features with the original DataFrame (excluding the original categorical features)

df = pd.concat([df.drop(columns=categorical_features), encoded_features], axis=1)


to_model = [
 'Overall MHQ',
 'MHQ_Sign',
 'Sudden or premature death of a loved one',
 'Divorce/separation  or family breakup',
 'Extreme poverty leading to homelessness and/or hunger.',
 'Forced family control over major life decisions (e.g. marriage)',
 'Prolonged sexual abuse| or severe sexual assault.',
 'Displacement from your home due to political| environmental or economic reasons',
 'Loss of your job or livelihood leading to an inability to make ends meet.',
 'Cyberbullying or online abuse',
 'Threatening| coercive or controlling behavior by another person',
 'Caring for a child or partner with a major chronic disability or illness',
 'I did not experience any of the above',
 'Involvement or close witness to a war',
 'Life threatening or debilitating injury or illness.',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster',
 'None of the above AT',
 'Sudden or premature death of a parent or sibling',
 'Prolonged emotional or psychological abuse or neglect from parent/caregiver',
 'Prolonged physical abuse| or severe physical assault CT',
 'Physical violence in the home between family members',
 'Prolonged or sustained bullying in person from peers',
 'Parental Divorce or family breakup',
 'Lived with a parent/caregiver who was an alcoholic or who regularly used street drugs',
 'Threatening| coercive or controlling behavior by another person CT',
 'I did not experience any of the above during my childhood',
 'Suffered a loss in a major fire| flood| earthquake| or natural disaster CT',
 'Displacement from your home due to political| environmental or economic reasons CT',
 'Life threatening or debilitating injury or illness CT',
 'Forced family control over major life decisions CT',
 'None of the above CT',
 'Tobacco products',
 'Alcoholic beverages',
 'Cannabis',
 'Vaping products',
 'Sedatives or Sleeping Pills',
 'Amphetamine type stimulants (e.g. speed| diet pills| ecstasy| etc.)',
 'Opioids',
 'Melatonina',
 'Cigarrillo',
 

 'Age_18',
 'Age_18-24',
 'Age_19',
 'Age_20',
 'Age_21-24',
 'Age_25-34',
 'Age_35-44',
 'Age_45-54',
 'Age_55-64',
 'Age_65-74',
 'Age_75-84',
 'Age_85+',

 'Biological Sex_Female',
 'Biological Sex_Male',
 'Biological Sex_Other/Intersex',
 'Biological Sex_Prefer not to say',
 
 'Frequency of getting a good nights sleep_All of the time',
 'Frequency of getting a good nights sleep_Hardly ever',
 'Frequency of getting a good nights sleep_Most days',
 'Frequency of getting a good nights sleep_Most of the time',
 'Frequency of getting a good nights sleep_Some of the time',
 
 
 'Frequency of doing exercise_Every day',
 'Frequency of doing exercise_Few days a week',
 'Frequency of doing exercise_Less than once a week',
 'Frequency of doing exercise_Once a week',
 'Frequency of doing exercise_Rarely/Never',
 #'Frequency of doing exercise_Several days a week',
 #'Frequency of doing exercise_Some days of the week',
 
 
 
 'Frequency of doing exercise_Every day',
 'Frequency of doing exercise_Few days a week',
 'Frequency of doing exercise_Less than once a week',
 'Frequency of doing exercise_Once a week',
 'Frequency of doing exercise_Rarely/Never',
 'Frequency of doing exercise_Several days a week',
 'Frequency of doing exercise_Some days of the week',
 
 'Processed food in diet_A few times a month',
 'Processed food in diet_A few times a week',
 'Processed food in diet_Once a day',
 'Processed food in diet_Rarely/Never',
 'Processed food in diet_Several times a day',

 'Frequency of Socializing_1-3 times a month',
 'Frequency of Socializing_Once a week',
 'Frequency of Socializing_Rarely/Never',
 'Frequency of Socializing_Several days a week']


df=df[to_model].copy()
print(len(df))





# %%


# %%


# %%


# %%
df=df_c

# %%



X = df.drop(columns=['MHQ_Sign','Overall MHQ'], axis=1)

# classification
y = df['MHQ_Sign']
# regressioin
y_regression = df['Overall MHQ']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.3, random_state=42)

# %%
X.to_csv("/Users/jerzybala/Desktop/data_causal_all_UPF_Age_18-24 and Anglosaxon_behavior_change.csv")

# %%
X.columns.tolist()

# %%
X['UPF_Social1'] = X['Processed food in diet_Rarely/Never'] + X['Frequency of Socializing_Once a week'] + X['Frequency of Socializing_Several days a week']


# %%


# %%


# %%
par1 = {
    'n_estimators': 200, 
    'learning_rate': 0.01,
    'max_depth': 10, 
    'min_child_weight': 1, 
    'gamma': 0.01, 
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Initialize XGBoost classifier and train

model_C = xgb.XGBClassifier(**par1)
model_C.fit(X_train, y_train)

# Predict and evaluate
predictions = model_C.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
auc = roc_auc_score(y_test, model_C.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"AUC: {auc}")


# %%
from sklearn.model_selection import GridSearchCV

# gs = GridSearchCV(XGBClassifier(), 
#                   param_grid, 
#                   scoring='accuracy',
#                   cv=5)
                  
# gs.fit(X_train, y_train)


# print(gs.best_params_)
# print(gs.best_score_)

# %%
# Initialize XGBoost regressor and train
#model = xgb.XGBRegressor()
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.3, random_state=42)

from sklearn.metrics import mean_absolute_error
#model_R = xgb.XGBRegressor()

model_R = xgb.XGBRegressor(
n_estimators=500,
learning_rate=0.1,
max_depth=6,
min_child_weight=1,
gamma=0.2,
subsample=0.4,
colsample_bytree=0.8,
#reg_alpha=10,
reg_lambda=0.1
)

model_R.fit(X_train_reg, y_train_reg)

# Separate predictions based on the sign of y_test
predictions = model_R.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, predictions)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_reg, predictions)
mae = mean_absolute_error(y_test_reg, predictions)

print('mae:', mae)
print('rmse:',rmse)
print('r2:', r2)


# %%
import shap

explainer_C = shap.Explainer(model_C)
shap_values_C = explainer_C(X_train)

# %%

shap_values_C_df = pd.DataFrame(shap_values_C.values, columns=shap_values_C.feature_names)
shap_values_C_df.head(4)


# %%


# %%
import shap
import matplotlib.pyplot as plt

# Specify the plot size directly in the SHAP summary_plot function
shap.summary_plot(shap_values_C, X_train, show=False, plot_size=(16,10), max_display=50)  # Adjust the size (width, height) as needed

plt.title(f'SHAP Summary Plot for Classification Age=18-24 and Core Anglosphere > UPF more Raerly')
plt.show()

# %%
# visualize the training set predictions
#  takes too long to run

import numpy as np

# Assuming shap_values_C is a numpy array
sampled_indices = np.random.choice(shap_values_C.shape[0], size=20000, replace=False)
sampled_shap_values_C = shap_values_C[sampled_indices]

shap.plots.force(sampled_shap_values_C)

# %% [markdown]
# 

# %%
# create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot('Processed food in diet_Rarely/Never', shap_values_C.values, X_train, interaction_index="Frequency of Socializing_Rarely/Never")


# %%
# summarize the effects of all the features
shap.plots.beeswarm(shap_values_C,max_display=15)

# %%
import shap
from shap import Explainer

shap.force_plot(
    shap.Explainer.expected_value[1], shap_values_C[1][:1000, :], X_train.iloc[:1000, :]
)

# %%


# %%


# %%


# %%


# %%
explainer_R = shap.Explainer(model_R)
shap_values_R = explainer_R(X_train_reg)


# %%
shap.plots.waterfall(shap_values[X_train])


# %%
# Specify the plot size directly in the SHAP summary_plot function
shap.summary_plot(shap_values_R, X_train, show=False, plot_size=(14,10), max_display=50)  # Adjust the size (width, height) as needed

plt.title(f'SHAP Summary Plot Regression Age=18-24 and and Core Anglosphere')
plt.show()

# %%


# %%


# %%


# %%


# %%
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# Assuming 'df' is your DataFrame with X and Y
# Replace 'df' with the name of your DataFrame
# Ensure your target column is named 'target'

def calculate_information_gain(X,y):
  
    # Calculating mutual information
    mutual_info = mutual_info_classif(X, y)

    # Creating a DataFrame for easier visualization
    info_gain_df = pd.DataFrame(mutual_info, index=X.columns, columns=['Information Gain'])
    
    # Sorting the DataFrame based on information gain
    sorted_info_gain = info_gain_df.sort_values(by='Information Gain', ascending=False)

    return sorted_info_gain

#Example usage:
sorted_info_gain = calculate_information_gain(X,y)
print(sorted_info_gain)


# %%
from tabulate import tabulate

#Example usage:
sorted_info_gain = calculate_information_gain(X,y)
print(tabulate(sorted_info_gain, headers='keys', tablefmt='psql'))


# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 


