# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:22:48 2024

@author: Aaron Stephenson
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

#File path
file_path = 'D:/ASU/DAT-490/Capstone/merged_play_by_play_positions.csv'
data = pd.read_csv(file_path)



print('/nBasic Summary: ', data.describe())

print('/nData Types: ', data.dtypes)

print('/nMissing Values: ', data.isnull().sum())

#Check for missing values in 'desc' column.
missing_desc = data['desc'].isna().sum()

#Extract rows where 'desc' mentions injury
injury_keywords = ['injury', 'injured', 'hurt']
injury_data = data[data['desc'].str.contains(' | '.join(injury_keywords), case = False, na = False)]

#Overview of the injury data
injury_data_overview = {
    "total_rows": len(data),
    "missing_desc": missing_desc,
    "injury_reports": len(injury_data)
    }

print(injury_data_overview, injury_data.head())

#Analyzing the position injury frequency
injury_by_position = injury_data['position'].value_counts(normalize = True)

#Creating a dataframe for better visualization
injury_by_position_df = injury_by_position.reset_index()
injury_by_position_df.columns = ['Position', 'Injury Count']

print(injury_by_position_df.head(10))

#Set style for the plot
sns.set(style = "whitegrid")

#Plot
plt.figure(figsize=(10,6))
sns.barplot(x= 'Injury Count', y = 'Position', data = injury_by_position_df, palette="Blues_d")

plt.title('NFL Injuries by Position')
plt.xlabel('Number of Injuries')
plt.ylabel('Position')
plt.show()

#%%
#Select relevant features for clustering
features = data[['position', 'play_type']]

#Adding injury frequency
data['injury_frequency'] = data['position'].map(injury_by_position)

#Add dummies of play types
data_dummies = pd.get_dummies(data, columns=['position', 'play_type'])

#Selecting cluster features
clustering_features = data_dummies[['injury_frequency'] + [col for col in data_dummies if col.startswith('play_type_')]]

#Fill missing values and scale
clustering_features = clustering_features.fillna(0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(clustering_features)

#Number of K clusters
number_of_clusters = 5

#K-means clustering
kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
clusters = kmeans.fit_predict(features_scaled)

#Add cluster labels based on original DF
data['cluster'] = clusters

#Analyze cluster centers
cluster_centers = kmeans.cluster_centers_

#Create DF for the cluster_centers
centers_df = pd.DataFrame(cluster_centers, columns=clustering_features.columns)

#Analyze dominant feature for each cluster
cluster_labels = []
for i in range(number_of_clusters):
    # Extracting only play type features for the current cluster center
   play_type_features = centers_df.iloc[i][[col for col in centers_df.columns if col.startswith('play_type_')]]
   # Identifying the dominant play type (the one with the highest value)
   dominant_play_type = play_type_features.idxmax()
   # Creating the label using the dominant play type
   label = f"Cluster {i}: {dominant_play_type}"
   cluster_labels.append(label)

#Display cluster labels
for label in cluster_labels:
    print(label)

#Plot positions distributed across
plt.figure(figsize=(10,6))
ax = sns.countplot(x='position', hue='cluster',data=data)
plt.title('Position Distribution Across Clusters')
plt.xlabel('Position')
plt.ylabel('Count')
legend_patches = [mpatches.Patch(color = ax.get_legend().legendHandles[i].get_facecolor(), label=cluster_labels[i]) for i in range(number_of_clusters)]
plt.legend(handles=legend_patches, title="Clusters", loc = "upper right")
plt.show()

#%%
#Identify non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

columns_to_drop = ['game_id', 'home_team', 'away_team', 'season_type', 'posteam', 'defteam', 
                   'side_of_field', 'game_date', 'game_half', 'time', 'start_time', 
                   'time_of_day', 'stadium', 'weather', 'location', 'roof', 'surface', 
                   'stadium_id', 'game_stadium', 'passer_id', 'rusher_id', 'receiver_id', 
                   'name', 'id', 'my_key_injury', 'key12', 'my_key']
data = data.drop(columns_to_drop, axis=1)

#encoding categorical variables
categorical_cols = ['play_type', 'position']
data = pd.get_dummies(data, columns=categorical_cols)

#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# Define your injury keywords
injury_keywords = ['injury', 'injured', 'hurt']

# Create a binary target variable
data['injury_occurred'] = data['desc'].str.contains('|'.join(injury_keywords), case=False, na=False).astype(int)

X = data.drop(['injury_occurred', 'desc'], axis = 1)
y= data['injury_occurred']

#Training and testing sets
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Replace infinite values and handle missing values
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

#Create and fit Random Forest
rf_model = RandomForestClassifier(random_state = 42)
rf_model.fit(X_train, y_train)

#Predictions and model eval
rf_predictions = rf_model.predict(X_test)
print(classification_report(y_test, rf_predictions))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

#%%
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV

# Assuming 'X' and 'y' are defined and ready for preprocessing and model fitting

# Create a pipeline that includes SMOTE, scaling, and the RandomForestClassifier
pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    # Add more parameters based on your dataset and model
}

# Initialize GridSearchCV with the pipeline
grid_search = GridSearchCV(estimator=pipeline, 
                           param_grid=param_grid, 
                           cv=3, n_jobs=1, scoring='roc_auc', verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate the best model found by GridSearchCV
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
print(classification_report(y_test, predictions))


#%%
"""from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE"""

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Impute missing values
imputer = SimpleImputer(strategy = 'mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

#Ensure no infinite values
X_train_imputed = np.nan_to_num(X_train_imputed, nan=np.nan, posinf=np.max(X_train_imputed), neginf=np.min(X_train_imputed))
X_test_imputed = np.nan_to_num(X_test_imputed, nan=np.nan, posinf=np.max(X_test_imputed), neginf=np.min(X_test_imputed))


#Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_imputed, y_train)

#It's important to apply scaling after SMOTE to ensure the synthetic samples are also scaled
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test_imputed)

#Fit the Random Forest model to the SMOTE-augmented training set
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train_scaled, y_train_smote)

"""X_train_imputed = np.nan_to_num(X_train_imputed, nan=np.nan)
X_test_imputed = np.nan_to_num(X_test_imputed, nan=np.nan)

# Then, reapply the imputer if necessary
X_train_imputed = imputer.fit_transform(X_train_imputed)
X_test_imputed = imputer.transform(X_test_imputed)"""

# Predictions and evaluation
predictions = rf_model.predict(X_test_scaled)
print(classification_report(y_test, predictions))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1]))

#%%

# Assuming 'best_model' is your trained RandomForest model from GridSearchCV
feature_importances = best_model.named_steps['classifier'].feature_importances_

# Assuming 'X_train' is your training dataset before applying SMOTE and scaling
# If you've applied transformations that alter the feature names, adjust accordingly
feature_names = X_train.columns

# Create a DataFrame to hold the feature names and their importance scores
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display the top 10 most important features
print(importances_df.head(10))

# Plotting feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df.head(20))
plt.title('Top 10 Feature Importances in RandomForest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

#%%

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#%%
TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")

from sklearn.metrics import classification_report

report = classification_report(y_test, predictions)
print(report)
#%%
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#%%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

precision, recall, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
average_precision = average_precision_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])

plt.figure()
plt.step(recall, precision, where='post', label='Precision-Recall curve (area = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()

#%%
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(rf_model, X_test_scaled, y_test)
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

#%%
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(rf_model, X_train_scaled, y_train_smote, cv=3)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show()