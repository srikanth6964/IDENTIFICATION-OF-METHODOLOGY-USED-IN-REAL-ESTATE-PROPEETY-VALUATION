import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Sample training data
X_train = np.array([
    [2012.917, 32.0, 84.87882, 10, 24.98298,121.54024],
    [2012.917, 19.5, 306.5947, 9, 24.98034,121.53951],
    [2013.583, 13.3, 561.9845, 5, 24.98746,121.54391],
    [2013.500, 13.3, 561.9845, 5, 24.98746,121.54391],
    [2013.000, 13.7, 4082.01500, 0,24.94155,121.50381]
])
y_train = np.array([37.9, 42.2, 47.3, 54.8, 15.1])

# Fit the StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)

# Fit the RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Save the scaler
with open('scale.pkl', 'wb') as f:
    pickle.dump(sc,f)

# Save the model
with open('price.pkl', 'wb') as f:
    pickle.dump(model,f)
                