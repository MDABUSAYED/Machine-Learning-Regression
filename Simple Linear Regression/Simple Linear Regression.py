
import graphlab

# # Load house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


sales = graphlab.SFrame('kc_house_data.gl/')
sales


# # Split data into training and testing

# We use seed=0 so that everyone running this notebook gets the same results.  In practice, you may set a random seed (or let GraphLab Create pick a random seed for you).  


train_data,test_data = sales.random_split(.8,seed=0)


# Let's compute the mean of the House Prices in King County in 2 different ways.
prices = sales['price'] # extract the price column of the sales SFrame -- this is now an SArray

# recall that the arithmetic average (the mean) is the sum of the prices divided by the total number of houses:
sum_prices = prices.sum()
num_houses = prices.size() # when prices is an SArray .size() returns its length
avg_price_1 = sum_prices/num_houses
avg_price_2 = prices.mean() # if you just want the average, the .mean() function
print "average price via method 1: " + str(avg_price_1)
print "average price via method 2: " + str(avg_price_2)


# As we see we get the same answer both ways


# if we want to multiply every price by 0.5 it's a simple as:
half_prices = 0.5*prices
# Let's compute the sum of squares of price. We can multiply two SArrays of the same length elementwise also with *
prices_squared = prices*prices
sum_prices_squared = prices_squared.sum() # price_squared is an SArray of the squares and we want to add them up.
print "the sum of price squared is: " + str(sum_prices_squared)



def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    
    
    # compute the product of the output and the input_feature and its sum
    
    # compute the squared value of the input_feature and its sum
    
    # use the formula for the slope
    
    # use the formula for the intercept
    
    numerator=(input_feature*output).mean()-(input_feature.mean())*(output.mean())
    denominator=((input_feature**2).mean())-(input_feature.mean())*(input_feature.mean())
    slope=numerator/denominator
    intercept=output.mean()-slope*input_feature.mean()
    return (intercept, slope)



# Now that we know it works let's build a regression model for predicting price based on sqft_living. Rembember that we train on train_data!


sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)


# # Predicting Values

# Now that we have the model parameters: intercept & slope we can make predictions. Using SArrays it's easy to multiply an SArray by a constant and add a constant value. Complete the following function to return the predicted output given the input_feature, slope and intercept:


def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    
    return intercept+slope*input_feature


# Now that we can calculate a prediction given the slope and intercept let's make a prediction. Use (or alter) the following to find out the estimated price for a house with 2650 squarefeet according to the squarefeet model we estiamted above.

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)


# # Residual Sum of Squares

# Now that we have a model and can make predictions let's evaluate our model using Residual Sum of Squares (RSS). Recall that RSS is the sum of the squares of the residuals and the residuals is just a fancy word for the difference between the predicted output and the true output. 
# 
# Complete the following (or write your own) function to compute the RSS of a simple linear regression model given the input_feature, output, intercept and slope:

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    prediction= intercept+slope*input_feature
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    RSS=sum((output-prediction)**2)
    # square the residuals and add them up

    return(RSS)


# Let's test our get_residual_sum_of_squares function by applying it to the test model where the data lie exactly on a line. Since they lie exactly on a line the residual sum of squares should be zero!

print get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope) # should be 0.0


# Now use your function to calculate the RSS on training data from the squarefeet model calculated above.


rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)




def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    
    return (output-intercept)/slope


# Now that we have a function to compute the squarefeet given the price from our simple regression model let's see how big we might expect a house that costs $800,000 to be.


my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)


# # New Model: estimate prices from bedrooms


# Estimate the slope and intercept for predicting 'price' based on 'bedrooms'
intercept, slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])
rss_prices_on_sqft = get_residual_sum_of_squares(train_data['bedrooms'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)

