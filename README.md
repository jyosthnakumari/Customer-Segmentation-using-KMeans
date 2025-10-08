# Customer-Segmentation-using-KMeans

## Project Overview
This project performs **customer segmentation** using **unsupervised machine learning** (K-Means Clustering) to group customers based on their **annual income** and **spending score**.  
The goal is to identify distinct customer groups, which can help businesses in **targeted marketing, personalized offers, and improving customer retention**.

---
# Customer-Segmentation-using-KMeans

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-wNJDvwfBB845XPRbBdMvRlzlX2yazhz#scrollTo=L2FAuLA8A0lh&printMode=true)

## Project Overview
This project performs **customer segmentation** using **unsupervised machine learning** (K-Means Clustering) to group customers based on their **annual income** and **spending score**.  
The goal is to identify distinct customer groups, which can help businesses in **targeted marketing, personalized offers, and improving customer retention**.

---

## Dataset
- **Name:** Mall Customers Dataset  
- **Source:** [Kaggle Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)  
- **Features:**
| Column | Description |
|--------|-------------|
| CustomerID | Unique identifier for each customer |
| Gender | Male or Female |
| Age | Age of the customer |
| Annual Income (k$) | Income of the customer in thousands |
| Spending Score (1-100) | Score assigned based on customer behavior |

---

## Project Steps

1. **Data Loading & Exploration**  
   - Loaded the dataset using Pandas  
   - Checked for missing values and data types  
   - Explored basic statistics  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized distribution of Age, Annual Income, and Spending Score  
   - Examined gender distribution  
   - Observed patterns in customer spending  

3. **Feature Selection & Scaling**  
   - Selected `Annual Income` and `Spending Score` as key features for clustering  
   - Standardized features using `StandardScaler`  

4. **Optimal Cluster Selection**  
   - Used the **Elbow Method** to determine the optimal number of clusters (`k=5`)  

5. **K-Means Clustering**  
   - Applied K-Means algorithm to create 5 customer clusters  
   - Assigned cluster labels to each customer  

6. **Cluster Visualization**  
   - Plotted clusters on a 2D scatter plot for interpretation  

7. **Insights & Analysis**  
   - Cluster 0 → High Income, High Spending  
   - Cluster 1 → Low Income, Low Spending  
   - Cluster 2 → Moderate Income, High Spending  
   - Cluster 3 → High Income, Low Spending  
   - Cluster 4 → Moderate Income, Moderate Spending  

---

## Technologies & Libraries
- Python 3  
- Pandas & NumPy (Data Handling)  
- Matplotlib & Seaborn (Data Visualization)  
- Scikit-learn (Machine Learning)  

---

## Outcome
- Identified 5 distinct customer segments  
- Visualized and interpreted customer groups for **business decision-making**  
- Exported clustered dataset for further analysis  

---

## Repository Contents
- `Customer_Segmentation.ipynb` → Full Jupyter/Colab notebook with EDA, clustering, and visualization  
- `Mall_Customers.csv` → Dataset used for the project  
- `Customer_Segments.csv` → Clustered output file  

---

## Future Enhancements
- Include additional features like customer purchase frequency, product preferences, or demographics  
- Deploy a **web dashboard** (Streamlit or Power BI) to make clusters interactive  
- Compare K-Means with other clustering algorithms (DBSCAN, Hierarchical Clustering)  

---

## License
This project is open-source and free to use for learning purposes.


## Dataset
- **Name:** Mall Customers Dataset  
- **Source:** [Kaggle Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)  
- **Features:**
| Column | Description |
|--------|-------------|
| CustomerID | Unique identifier for each customer |
| Gender | Male or Female |
| Age | Age of the customer |
| Annual Income (k$) | Income of the customer in thousands |
| Spending Score (1-100) | Score assigned based on customer behavior |

---

## Project Steps

1. **Data Loading & Exploration**  
   - Loaded the dataset using Pandas  
   - Checked for missing values and data types  
   - Explored basic statistics  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized distribution of Age, Annual Income, and Spending Score  
   - Examined gender distribution  
   - Observed patterns in customer spending  

3. **Feature Selection & Scaling**  
   - Selected `Annual Income` and `Spending Score` as key features for clustering  
   - Standardized features using `StandardScaler`  

4. **Optimal Cluster Selection**  
   - Used the **Elbow Method** to determine the optimal number of clusters (`k=5`)  

5. **K-Means Clustering**  
   - Applied K-Means algorithm to create 5 customer clusters  
   - Assigned cluster labels to each customer  

6. **Cluster Visualization**  
   - Plotted clusters on a 2D scatter plot for interpretation  

7. **Insights & Analysis**  
   - Cluster 0 → High Income, High Spending  
   - Cluster 1 → Low Income, Low Spending  
   - Cluster 2 → Moderate Income, High Spending  
   - Cluster 3 → High Income, Low Spending  
   - Cluster 4 → Moderate Income, Moderate Spending  

---

## Technologies & Libraries
- Python 3  
- Pandas & NumPy (Data Handling)  
- Matplotlib & Seaborn (Data Visualization)  
- Scikit-learn (Machine Learning)  

---

## Outcome
- Identified 5 distinct customer segments  
- Visualized and interpreted customer groups for **business decision-making**  
- Exported clustered dataset for further analysis  

---

## Repository Contents
- `Customer_Segmentation.ipynb` → Full Jupyter/Colab notebook with EDA, clustering, and visualization  
- `Mall_Customers.csv` → Dataset used for the project  
- `Customer_Segments.csv` → Clustered output file  

---

## Future Enhancements
- Include additional features like customer purchase frequency, product preferences, or demographics  
- Deploy a **web dashboard** (Streamlit or Power BI) to make clusters interactive  
- Compare K-Means with other clustering algorithms (DBSCAN, Hierarchical Clustering)  

---

## License
This project is open-source and free to use for learning purposes.
