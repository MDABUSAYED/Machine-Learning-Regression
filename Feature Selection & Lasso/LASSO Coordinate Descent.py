


import graphlab
import numpy as np
import math


# # Load in house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.



sales = graphlab.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int) 



 # note this allows us to refer to numpy as np instead 



def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe['price']
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)


# Also, copy and paste the `predict_output()` function to compute the predictions for an entire matrix of features given the matrix and the weights:


def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


# # Normalize features
# In the house dataset, features vary wildly in their relative magnitude: `sqft_living` is very large overall compared to `bedrooms`, for instance. As a result, weight for `sqft_living` would be much smaller than weight for `bedrooms`. This is problematic because "small" weights are dropped first as `l1_penalty` goes up. 
# 
# To give equal considerations for all features, we need to **normalize features** as discussed in the lectures: we divide each feature by its 2-norm so that the transformed feature has norm 1.
# 
# Let's see how we can do this normalization easily with Numpy: let us first consider a small matrix.





# Using the shorthand we just covered, write a short function called `normalize_features(feature_matrix)`, which normalizes columns of a given feature matrix. The function should return a pair `(normalized_features, norms)`, where the second item contains the norms of original features. As discussed in the lectures, we will use these norms to normalize the test data in the same way as we normalized the training data. 



def normalize_features(x):
    norms=np.linalg.norm(x, axis=0)
    return x/norms,norms



simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)


simple_feature_matrix, norms = normalize_features(simple_feature_matrix)




prediction =  predict_output(simple_feature_matrix, weights)





# So we can say that `ro[i]` quantifies the significance of the i-th feature: the larger `ro[i]` is, the more likely it is for the i-th feature to be retained.

# ## Single Coordinate Descent Step

# Using the formula above, implement coordinate descent that minimizes the cost function over a single feature i. Note that the intercept (weight 0) is not regularized. The function should accept feature matrix, output, current weights, l1 penalty, and index of feature to optimize over. The function should return new weight for feature i.


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction =predict_output(feature_matrix, weights)   
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.sum(feature_matrix[:,i]*(output - prediction + weights[i]*feature_matrix[:,i]))

    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + (l1_penalty/2.)
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - (l1_penalty/2)
    else:
        new_weight_i = 0.
    
    return new_weight_i

print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]), 
                                   np.array([1., 1.]), np.array([1., 4.]), 0.1)




def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = initial_weights.copy()    
    # converged condition variable    
    converged  = False        
    while not converged:         
        max_change = 0
        for i in range(len(weights)):
            old_weights_i = weights[i] 
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)                        
            change_i = np.abs(old_weights_i - weights[i])             
            if change_i > max_change:                
                max_change = change_i        
        if max_change < tolerance:              
            converged = True     
    return weights



(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features


# Then, run your implementation of LASSO coordinate descent:


weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
weights
predictions = predict_output(normalized_simple_feature_matrix, weights)
# rss = 0
# for i in range(0, len(predictions)):
error = predictions - sales['price']
rss = (error *error).sum()
print rss




train_data,test_data = sales.random_split(.8,seed=0)




all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']


# First, create a normalized feature matrix from the TRAINING data with these features.  (Make you store the norms for the normalization, since we'll use them later)


(all_feature_matrix, output) = get_numpy_data(train_data, all_features, my_output)
(normalized_all_feature_matrix, simple_norms) = normalize_features(all_feature_matrix) # normalize features
my_output = 'price'
initial_weights = np.zeros(14)
l1_penalty = 1e7
tolerance = 1.0


# First, learn the weights with `l1_penalty=1e7`, on the training data. Initialize weights to all zeros, and set the `tolerance=1`.  Call resulting weights `weights1e7`, you will need them later.


weights7 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty=1e7, tolerance=1)
print weights7


weights8 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty=1e8, tolerance=1)
print weights8




weights4 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty=1e4, tolerance=1)
print weights4



normalized_weights7 = weights7 / simple_norms
print normalized_weights7[3]
normalized_weights4 = weights4 / simple_norms
normalized_weights8 = weights8 / simple_norms




(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')


# Compute the RSS of each of the three normalized weights on the (unnormalized) `test_feature_matrix`:


prediction =  predict_output(test_feature_matrix, normalized_weights4)
error = prediction -test_data['price']
rss = (error *error).sum()
print rss



prediction =  predict_output(test_feature_matrix, normalized_weights7)
error = prediction -test_data['price']
rss = (error *error).sum()
print rss



prediction =  predict_output(test_feature_matrix, normalized_weights8)
error = prediction -test_data['price']
rss = (error *error).sum()
print rss




rho=[79400299.034929171, 87939466.772991076, 80966697.675965667]




penalties =[1.4e8, 1.64e8, 1.73e8, 1.9e8, 2.3e8]
for i in  range(len(penalties)):
        test1 =penalties[i]/2 < rho
        test2 = rho < penalties[i]/2
        if (test1[3]==1 & test2[3]==1):
                if (test1[2]==0 | test2[2]==0):
                        print(paste(penalties[i]," acheives the effect"))






