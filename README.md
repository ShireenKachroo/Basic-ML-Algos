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
