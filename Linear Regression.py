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


train = pd.read_csv("Linear Regression/trainData.csv")

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

# gradient Descent
def cost_func(X, y, theta):
    m = len(y)
    h = X @ theta
    J = (1 / (2 * m)) * sum((h - y) ** 2)
    return J
#gradient descent
def gradient_descent(X, y, theta, alpha, iteration):
    m = len(y)
    cost_history = []
    
    for i in range(iteration):
        h = X @ theta
        loss = h - y
        gradient = [0] * len(theta)
        
        for j in range(len(theta)):
            gradient[j] = sum(loss * X.iloc[:, j]) / m
            
        for j in range(len(theta)):
            theta[j] -= alpha * gradient[j]
        
        cost_ = cost_func(X, y, theta)
        cost_history.append(cost_)
    
    return theta, cost_history

# initialization
theta_initial = [0] * X.shape[1]

# alpha values and iteration
alpha_values = [0.1, 0.01, 0.3, 0.03]
num_iterations = 1000

for alpha in alpha_values:
    theta = theta_initial.copy()
    theta, cost_history = gradient_descent(X, price, theta, alpha, num_iterations)
    
    plt.plot(range(len(cost_history)), cost_history, label=f'Alpha = {alpha}')

# plotting
plt.xlabel('Iterations')
plt.ylabel('Cost J')
plt.title('Cost values')
plt.legend()
plt.show()



