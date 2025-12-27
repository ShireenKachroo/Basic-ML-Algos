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



