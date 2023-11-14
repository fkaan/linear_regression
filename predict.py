import pandas as pd
import matplotlib.pyplot as plt

# scaling 
def min_max_scaling(feature):
    return (feature - feature.min()) / (feature.max() - feature.min())

# feature matrix function
def feature_matrix(year, km_driven, fuel_type, seller_type, transmission, owner):
   
    scaled_year = min_max_scaling(year) #minmax for numerical features
    scaled_km_driven = min_max_scaling(km_driven)
    
    # one-hot encoding for categorical features
    encoded_fuel = pd.get_dummies(fuel_type, prefix='fuel')
    encoded_seller = pd.get_dummies(seller_type, prefix='seller')
    encoded_transmission = pd.get_dummies(transmission, prefix='transmission')
    encoded_owner = pd.get_dummies(owner, prefix='owner')
    
    # features into a single dataframe
    feature_matrix = pd.concat([scaled_year, scaled_km_driven, encoded_fuel, encoded_seller, encoded_transmission, encoded_owner], axis=1)
    
    # Insert a column of 1s for the intercept term
    feature_matrix.insert(0, 'intercept', 1)
    
    return feature_matrix


train = pd.read_csv("/trainData.csv")

# features 
year = train["year"]
km_driven = train["km_driven"]
fuel_type = train["fuel"]
seller_type = train["seller_type"]
transmission = train["transmission"]
owner = train["owner"]
price = train["selling_price"]

# feature matrix
X = feature_matrix(year, km_driven, fuel_type, seller_type, transmission, owner)


# cost
def compute_cost(X, y, theta):
    m = len(y)
    error = X.dot(theta) - y
    cost = (1/(2*m)) * sum(error**2)
    return cost
# gradient descent
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        error = X.dot(theta) - y
        theta = theta - (learning_rate/m) * X.T.dot(error)
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# initialization
theta_initial = [0] * X.shape[1]

# target variable
y = train["selling_price"]

learning_rate = 0.3
num_iterations = 1000

# initialize weights
theta = pd.Series(0, index=X.columns)

# running gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)


# save the results to an Excel file
result_df = pd.DataFrame({'Actual_Price': y, 'Predicted_Price': X.dot(theta)})

result_df.to_csv(r"Linear Regression\predicted.csv", index = False)# Update the path
