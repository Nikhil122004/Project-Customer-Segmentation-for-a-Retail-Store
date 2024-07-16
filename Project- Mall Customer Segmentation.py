import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'Mall_Customers.csv'
data = pd.read_csv(file_path)


# Display the first few rows of the dataset
print(data.head(10))

# Counting the null values
count=data.isnull().sum()
print("Null values:",count)


#Replacing if the Null values in Age column with mean
"""mean_age=data['Age'].mean()

data["Age"].fillna(mean_age,inplace=True)"""

# Renaming columns for better readability
if len(data.columns) == 5:
    data.columns = ["CustomerID", "Gender", "Age", "Annual_Income", "SpendingScore"]
else:
    print("The number of column names provided does not match the number of columns in the DataFrame.")
print(data.head(10))


#Replacing if the Null values in Gender column with mode
"""mode_gender=data['Gender'].mode()[0]
type(mode_gender)
mode_gender

data.dropna(inplace=True)

data["Gender"].fillna(mode_gender,inplace=True)"""


# Data transformation (e.g., encoding categorical variables)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

count=data.isnull().sum()
#print(count)


#Exploratory Data Analysis (EDA)
print(data.describe())


#histplot for Age Distribution
# Visualizing distributions

# Setting the aesthetic style of the plots
sns.set_style("whitegrid")

# Creating the plot
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True, color='red')

# Adding titles and labels
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Displaying the plot
plt.show()

#histplot for Annual_Income Distribution

# Setting the aesthetic style of the plots
sns.set_style("whitegrid")

# Creating the plot
plt.figure(figsize=(10, 6))
sns.histplot(data['Annual_Income'], bins=30, kde=True,color='green')

# Adding titles and labels
plt.title('Annual Income Distribution',fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

#Displaying the plot
plt.show()

#histplot for SpendingScore Distribution

# Setting the aesthetic style of the plots
sns.set_style("whitegrid")

# Creating the plot
plt.figure(figsize=(10, 6))
sns.histplot(data['SpendingScore'], bins=30, kde=True)

# Adding titles and labels
plt.title('SpendingScore Distribution',fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

#Displaying the plot
plt.show()


# Visualizing Relationships

# Setting the aesthetic style of the plots
sns.set_style("whitegrid")

# Creating the plot
plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(data=data, x='Annual_Income', y='SpendingScore', hue='Gender', style='Gender', s=100, palette='viridis')

# Adding titles and labels
plt.title('Income vs Spending Score', fontsize=16)
plt.xlabel('Annual_Income', fontsize=14)
plt.ylabel('Spending Score', fontsize=14)

# Customizing the legend
plt.legend(title='Gender', title_fontsize='13', fontsize='11')

# Displaying the plot
plt.show()

# Bar Plot for gender count
# Setting the aesthetic style of the plots
sns.set_style("whitegrid")

# Creating the bar plot
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Gender', palette='viridis')
plt.title('Gender Distribution', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

#Box Plot
# Creating the box plot for Age by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Gender', y='Age', palette='pastel')
plt.title('Age Distribution by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.show()



# Feature selection
features = data[['Age', 'Annual_Income', 'SpendingScore']]

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Applying K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Evaluating cluster quality
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual_Income', y='SpendingScore', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.show()

