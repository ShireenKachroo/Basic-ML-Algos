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


