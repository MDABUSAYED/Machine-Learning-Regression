

import graphlab

from math import sqrt

# # Load in house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.



sales = graphlab.SFrame('kc_house_data.gl/')
sales


# If we want to do any "feature engineering" like creating new features or adjusting existing ones we should do this directly using the SFrames as seen in the other Week 2 notebook. For this notebook, however, we will work with the existing features.

# # Convert to Numpy Array

# Although SFrames offer a number of benefits to users (especially when using Big Data and built-in graphlab functions) in order to understand the details of the implementation of algorithms it's important to work with a library that allows for direct (and optimized) matrix operations. Numpy is a Python solution to work with matrices (or any multi-dimensional "array").
# 
# Recall that the predicted value given the weights and the features is just the dot product between the feature and weight vector. Similarly, if we put all of the features row-by-row in a matrix then the predicted value for *all* the observations can be computed by right multiplying the "feature matrix" by the "weight vector". 
# 
# First we need to take the SFrame of our data and convert it into a 2D numpy array (also called a matrix). To do this we use graphlab's built in .to_dataframe() which converts the SFrame into a Pandas (another python library) dataframe. We can then use Panda's .as_matrix() to convert the dataframe into a numpy matrix.


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



def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


# If you want to test your code run the following cell:


test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0



def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2*np.dot(errors, feature)
    return(derivative)


# To test your feature derivartive run the following:


# # Gradient Descent

# Now we will write a function that performs a gradient descent. The basic premise is simple. Given a starting point we update the current weights by moving in the negative gradient direction. Recall that the gradient is the direction of *increase* and therefore the negative gradient is the direction of *decrease* and we're trying to *minimize* a cost function. 
# 
# The amount by which we move in the negative gradient *direction*  is called the 'step size'. We stop when we are 'sufficiently close' to the optimum. We define this by requiring that the magnitude (length) of the gradient vector to be smaller than a fixed 'tolerance'.
# 
# With this in mind, complete the following gradient descent function below using your derivative function above. For each step in the gradient descent we update the weight for each feature befofe computing our stopping criteria

 # recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)



def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            drivative = feature_derivative(errors, feature_matrix[:, i])
            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares= np.sum( drivative* drivative)
            #print np.sum( drivative* drivative)
            #print gradient_sum_squares
            
            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size * drivative
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)


# A few things to note before we run the gradient descent. Since the gradient is a sum over all the data points and involves a product of an error and a feature the gradient itself will be very large since the features are large (squarefeet) and the output is large (prices). So while you might expect "tolerance" to be small, small is only relative to the size of the features. 
# 
# For similar reasons the step size will be much smaller than you might expect but this is because the gradient has such large values.

# # Running the Gradient Descent as Simple Regression

# First let's split the data into training and test data.


train_data,test_data = sales.random_split(.8,seed=0)


# Although the gradient descent is designed for multiple regression since the constant is now a feature we can use the gradient descent function to estimat the parameters in the simple regression on squarefeet. The folowing cell sets up the feature_matrix, output, initial weights and step size for the first model:


# let's test out the gradient descent
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7


# Next run your gradient descent with the above parameters.



weights= regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
print weights




(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)


# Now compute your predictions using test_simple_feature_matrix and your weights from above.


predictions=  predict_output(test_simple_feature_matrix, weights)


print  predictions[0]




a=predictions - test_data['price']
a=np.array(a)
rss= np.dot(a,a)
print rss


# # Running a multiple regression

# Now we will use more than one actual feature. Use the following code to produce the weights for a second model with the following parameters:


model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9


# Use the above parameters to estimate the model weights. Record these values for your quiz.

simple_weights = regression_gradient_descent(feature_matrix, output,initial_weights, step_size, tolerance)
print simple_weights


# Use your newly estimated weights and the predict_output function to compute the predictions on the TEST data. Don't forget to create a numpy array for these features from the test set first!


(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)




test_predictions = predict_output(test_simple_feature_matrix, simple_weights)





test_predictions[0]




a=test_predictions - test_data['price']
a=np.array(a)
rss= np.dot(a,a)
print rss


