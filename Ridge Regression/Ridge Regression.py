

import graphlab


# # Load in house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


sales = graphlab.SFrame('kc_house_data.gl/')


# If we want to do any "feature engineering" like creating new features or adjusting existing ones we should do this directly using the SFrames as seen in the first notebook of Week 2. For this notebook, however, we will work with the existing features.

# # Import useful functions from previous notebook

# As in Week 2, we convert the SFrame into a 2D Numpy array. Copy and paste `get_numpy_data()` from the second notebook of Week 2.


import numpy as np # note this allows us to refer to numpy as np instead 



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



def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if(feature_is_constant):
        derivative = 2*np.dot(errors,feature)
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        derivative = 2*np.dot(errors,feature) + 2*l2_penalty*weight
    return derivative


# To test your feature derivartive run the following:




# # Gradient Descent

# Now we will write a function that performs a gradient descent. The basic premise is simple. Given a starting point we update the current weights by moving in the negative gradient direction. Recall that the gradient is the direction of *increase* and therefore the negative gradient is the direction of *decrease* and we're trying to *minimize* a cost function. 
# 
# The amount by which we move in the negative gradient *direction*  is called the 'step size'. We stop when we are 'sufficiently close' to the optimum. Unlike in Week 2, this time we will set a **maximum number of iterations** and take gradient steps until we reach this maximum number. If no maximum number is supplied, the maximum should be set 100 by default. (Use default parameter values in Python.)
# 
# With this in mind, complete the following gradient descent function below using your derivative function above. For each step in the gradient descent, we update the weight for each feature before computing our stopping criteria.


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights) # make sure it's a numpy array
    i=0
    #while not reached maximum number of iterations:
    while i< max_iterations:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights) 
        # compute the errors as predictions - output
        errors = predictions - output
        for j in xrange(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            if j == 0:
                derivative = feature_derivative_ridge(errors, feature_matrix[:,0], weights[0], l2_penalty, True)
            else:
                derivative = feature_derivative_ridge(errors, feature_matrix[:,j], weights[j], l2_penalty, False) 
                weights[j] = weights[j] - step_size*derivative
            # subtract the step size times the derivative from the current weight
            i=i+1
    return weights


# # Visualizing effect of L2 penalty

# The L2 penalty gets its name because it causes weights to have small L2 norms than otherwise. Let's see how large weights get penalized. Let us consider a simple model with 1 feature:


simple_features = ['sqft_living']
my_output = 'price'


# Let us split the dataset into training set and test set. Make sure to use `seed=0`:

train_data,test_data = sales.random_split(.8,seed=0)


# In this part, we will only use `'sqft_living'` to predict `'price'`. Use the `get_numpy_data` function to get a Numpy versions of your data with only this feature, for both the `train_data` and the `test_data`. 


(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)


# Let's set the parameters for our optimization:


initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000


# First, let's consider no regularization.  Set the `l2_penalty` to `0.0` and run your ridge regression algorithm to learn the weights of your model.  Call your weights:
# 
# `simple_weights_0_penalty`
# 
# we'll use them later.


l2_penalty = 0.0
simple_weights_0_penalty=  ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print simple_weights_0_penalty


# Next, let's consider high regularization.  Set the `l2_penalty` to `1e11` and run your ridge regression algorithm to learn the weights of your model.  Call your weights:
# 
# `simple_weights_high_penalty`
# 
# we'll use them later.


l2_penalty = 1e11
simple_weights_high_penalty=  ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print simple_weights_high_penalty


# This code will plot the two learned models.  (The blue line is for the model with no regularization and the red line is for the one with high regularization.)


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')


# Compute the RSS on the TEST data for the following three sets of weights:
# 1. The initial weights (all zeros)
# 2. The weights learned with no regularization
# 3. The weights learned with high regularization
# 
# Which weights perform best?


train_data,test_data = sales.random_split(.8,seed=0)
model_features = ['sqft_living']
my_output = 'price'
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
test_predictions = predict_output(test_simple_feature_matrix,[0.,0.])
rss = 0
for i in range(0, len(test_predictions)):
    error = test_predictions[i] - test_data['price'][i]
    rss += error * error
print rss




train_data,test_data = sales.random_split(.8,seed=0)
model_features = ['sqft_living']
my_output = 'price'
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
test_predictions = predict_output(test_simple_feature_matrix,simple_weights_0_penalty)
rss = 0
for i in range(0, len(test_predictions)):
    error = test_predictions[i] - test_data['price'][i]
    rss += error * error
print rss




train_data,test_data = sales.random_split(.8,seed=0)
model_features = ['sqft_living']
my_output = 'price'
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
test_predictions = predict_output(test_simple_feature_matrix,simple_weights_high_penalty)
rss = 0
for i in range(0, len(test_predictions)):
    error = test_predictions[i] - test_data['price'][i]
    rss += error * error
print rss



model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)


# We need to re-inialize the weights, since we have one extra parameter. Let us also set the step size and maximum number of iterations.


initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000


# First, let's consider no regularization.  Set the `l2_penalty` to `0.0` and run your ridge regression algorithm to learn the weights of your model.  Call your weights:
# 
# `multiple_weights_0_penalty`


l2_penalty = 0.0
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print multiple_weights_0_penalty


# Next, let's consider high regularization.  Set the `l2_penalty` to `1e11` and run your ridge regression algorithm to learn the weights of your model.  Call your weights:
# 
# `multiple_weights_high_penalty`


l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print multiple_weights_high_penalty


train_data,test_data = sales.random_split(.8,seed=0)
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
test_predictions = predict_output(test_simple_feature_matrix,multiple_weights_0_penalty)
rss = 0
for i in range(0, len(test_predictions)):
    error = test_predictions[i] - test_data['price'][i]
    rss += error * error
print rss


train_data,test_data = sales.random_split(.8,seed=0)
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
test_prediction1 = predict_output(test_simple_feature_matrix[0,:],multiple_weights_0_penalty)
test_prediction2 = predict_output(test_simple_feature_matrix[0,:],multiple_weights_high_penalty)
print test_simple_feature_matrix[0,:]
print  test_output[0]
print test_prediction1
print test_prediction2
print multiple_weights_0_penalty
print multiple_weights_high_penalty
print test_prediction1-test_output[0]
print test_prediction2-test_output[0]

