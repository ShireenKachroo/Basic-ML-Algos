# Basic-ML-Algos

## 1. LINEAR REGRESSION

<b> When to use: </b> For regression tasks. It gives continuous outputs, so best to use when the predicted value is a continuous numeric variable.<br>

<b> Formula: </b> y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ <br>
It fits a straight line (or hyperplane) that minimizes squared error between predicted and actual values.<br>

<b> Pros: </b> 
- Simple to understand and implement  
- Coefficients show direct impact of features on target  
- Works well with small datasets  
- Computationally inexpensive  
- Easy to interpret and explain
   
<b> Cons: </b> 
- Assumes linear relationship between variables  
- Sensitive to outliers  
- Assumes constant variance of errors (homoscedasticity)  
- Struggles with multicollinearity  
- Not suitable for classification tasks
  
<b> When not to use: </b> 
- Non-linear relationships  
- Many categorical variables without encoding  
- Discrete target variables (classification problems)

## 2. LOGISTIC REGRESSION  

<b> When to use: </b> For **classification tasks**. It predicts the probability of belonging to a class (e.g., 0 or 1) and is best used when the target variable is categorical.<br>  

<b> Formula: </b>  
P(y=1|x) = 1/{1 + e^(-z)}  where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ 
<br>It applies the **sigmoid (logistic) function** to map linear combinations of inputs into probabilities between 0 and 1.<br>  

<b> Pros: </b>  
- Simple and widely used for binary classification  
- Outputs probabilities, not just class labels   
- Works well with small to medium datasets  
- Computationally efficient and easy to implement  
- Can be extended to multi-class classification (ordinal, Multinomial)<br>  

<b> Cons: </b>  
- Assumes linear relationship between features and log-odds of the outcome  
- Sensitive to multicollinearity among predictors  
- Struggles with non-linear decision boundaries unless features are engineered  
- Not ideal for very large feature spaces without regularization  
- Performance may degrade with highly imbalanced datasets<br>  

<b> When not to use: </b>  
- Complex non-linear classification problems
- Datasets with many categorical variables that aren’t properly encoded  
- Situations where interpretability of coefficients is not useful or necessary  
- Regression tasks 

## 3. DECISION TREE  

<b> When to use: </b> For both **classification and regression tasks**. Decision Trees split data into branches based on feature values, making them useful when relationships between variables are non-linear or when interpretability is important.<br>  

<b> Pros: </b>  
- Easy to understand and visualize (tree-like structure)  
- Handles both numerical and categorical data without much preprocessing  
- Captures non-linear relationships effectively  
- Requires little data preparation (no need for scaling/normalization)  
- Can handle multi-output problems  
- Works well as a base learner in ensemble methods<br>  

<b> Cons: </b>  
- Prone to **overfitting** if not pruned or regularized  
- Small changes in data can lead to very different trees (high variance)  
- Less accurate compared to ensemble methods  
- Can create biased trees if class imbalance is not handled  
- Not ideal for very high-dimensional datasets without feature selection<br>  

<b> When not to use: </b>  
- Very large datasets where ensemble methods perform better  
- Situations requiring smooth predictions (trees create step-like boundaries)  
- Highly imbalanced datasets without proper handling 
- Tasks where stability of the model is critical

### 4. RANDOM FOREST
<b> When to use: </b>
For both classification and regression tasks. Random Forest is an ensemble method that builds multiple decision trees and combines their outputs. Best used when you want higher accuracy and robustness compared to a single decision tree.<br>

<b> Pros: </b>
- Reduces overfitting compared to a single decision tree
- Handles both numerical and categorical features well
- Works well with large datasets and high-dimensional feature spaces
- Robust to noise and outliers
- Provides feature importance ranking
- Can handle missing values effectively<br>

<b> Cons: </b>
- Less interpretable compared to a single decision tree (black-box nature)
- Computationally expensive with many trees
- Slower predictions compared to simpler models
- Large memory usage for big forests
- May still struggle with extreme class imbalance if not addressed<br>

<b> When not to use: </b>
- When interpretability is crucial 
- Real-time applications requiring very fast predictions
- Very small datasets
- Highly imbalanced datasets without resampling or class-weight adjustments

### 5. KNN Algorithm
<b> When to use: </b>
For both classification and regression tasks. KNN is a lazy learning algorithm that makes predictions based on the majority class (classification) or average value (regression) of the k nearest data points in the feature space. Best used when decision boundaries are irregular and data is not linearly separable.<br>

<b> Pros: </b>
- Simple and intuitive to understand
- Naturally handles multi-class classification
- Flexible: works for both classification and regression
- Captures complex decision boundaries effectively
  
<b> Cons: </b>
- Computationally expensive for large datasets
- Sensitive to irrelevant or redundant features
- Requires feature scaling (normalization/standardization) for meaningful distance comparisons
- Performance depends heavily on choice of k and distance metric
- Struggles with high-dimensional data
- Can be biased if class imbalance is not handled properly
  
<b> When not to use: </b>
- Very large datasets 
- High-dimensional feature spaces without dimensionality reduction
- Datasets with many irrelevant features or noise
- Highly imbalanced datasets without resampling or weighted voting
- Real-time applications requiring very fast predictions


## 6. Naïve Bayes

<b> When to use: </b>  
For **classification tasks**, especially text classification (spam detection, sentiment analysis, document categorization). Best used when features are conditionally independent given the class.<br>  

<b> Formula: </b>  
Bayes’ Theorem:  
P(C|X) = [P(X|C) * P(C)] / P(X)  

<b> Pros: </b>  
- Very fast and efficient for large datasets  
- Performs well in text classification and NLP tasks  
- Requires small amount of training data  
- Handles multi-class problems naturally  
- Easy to implement and interpret  

<b> Cons: </b>  
- Assumes independence between features 
- Struggles with correlated predictors  
- Continuous variables require distribution assumptions
- Can be biased if class imbalance is not handled  

<b> When not to use: </b>  
- Datasets with highly correlated features  
- Complex feature interactions where independence assumption fails  
- Regression tasks  
- Small datasets with skewed class distributions
  

## 7. Support Vector Machine (SVM)

<b> When to use: </b>  
For **classification tasks** (and regression via SVR). Best used when data is high-dimensional and decision boundaries are complex. Works well for text classification, image recognition, and bioinformatics.<br>  

<b> Pros: </b>  
- Effective in high-dimensional spaces  
- Works well with clear margin of separation  
- Flexible with kernel functions for non-linear boundaries  
- Robust to overfitting in moderate dimensions  
- Can handle both classification and regression  

<b> Cons: </b>  
- Computationally expensive for very large datasets  
- Requires careful tuning of kernel and parameters
- Less interpretable compared to simpler models  
- Sensitive to choice of kernel  
- Struggles with noisy datasets and overlapping classes  

<b> When not to use: </b>  
- Very large datasets (training time is slow)  
- Problems requiring high interpretability  
- Datasets with heavy noise or overlapping classes  
- When quick predictions are needed in real-time systems

## 8. K-Means Clustering

<b> When to use: </b>  
For **unsupervised clustering** to group similar data points. Best used when clusters are roughly spherical, well-separated, and features are continuous and scaled. Common in customer segmentation, anomaly detection, and compressing/high-level summaries of data.<br>  

<b> Pros: </b>  
- Simple, fast, and scalable for large datasets  
- Easy to implement and understand  
- Works well when clusters are compact and similar in size  
- Efficient with k-means++ initialization  
- Compatible with many real-world preprocessing pipelines  

<b> Cons: </b>  
- Requires choosing the number of clusters (k)  
- Sensitive to feature scaling and outliers  
- Assumes spherical, equally sized clusters  
- Can converge to local minima depending on initialization  
- Struggles with non-convex or overlapping clusters  

<b> When not to use: </b>  
- highly overlapping clusters  
- Datasets with significant outliers or unscaled features  
- Categorical-only data without proper encoding  
- When interpretability of clusters is critical and domain context is weak  
- When the number of clusters is unknown and hard to estimate (consider Elbow/Silhouette methods first)


