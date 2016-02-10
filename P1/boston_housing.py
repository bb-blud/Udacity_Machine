"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

################################
### ADD EXTRA LIBRARIES HERE ###
################################
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import make_scorer
#from sklearn.metrics import median_absolute_error as metric
import sklearn.metrics as m

def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    # Please calculate the following values using the Numpy library
    # Size of data (number of houses)?
    print "Number of houses: ", housing_features.shape[0]
    # Number of features?
    print "Features per house: ", housing_features.shape[1]
    # Minimum price?
    print "Minimum price: ", np.min(housing_prices)
    # Maximum price?
    print "Maximum price: ", np.max(housing_prices)
    # Calculate mean price?
    print "Mean price: ", np.mean(housing_prices)
    # Calculate median price?
    print "Median price: ", np.median(housing_prices)
    # Calculate standard deviation?
    print "Standard deviation: ", np.std(housing_prices)


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################
    #The cross validation train_test split is the first step in constructing all the model diagnostics we perform in the code
    #below. Without test data, we would not be able to gauge whether the model is tending to overfit the data,
    #A model can be made to fit a certain data set exactly, but this does not guarantee it will generalize well to yet unseen data. 
    #We can spot patterns that indicate over fitting in training vs test error plots for our model.  
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.3, random_state = 0)

    return X_train, y_train, X_test, y_test


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################
    #I don't know how legitimate this justification is but I settled on median_absolute_error as a metric because it was providing
    #smoother error curves than mse and mean_ae. This can be partially expected because of med_ae's resistance to outliers. 
    #The choice for the optimal max_depth parameter was more apparent in the error curves because of their less erratic behavior. Further 
    #the med_ae choice agrees with some computations I did using grid search further down. 
    return m.median_absolute_error(label, prediction)
#    return metric(label,prediction)
    # # The following page has a table of scoring functions in sklearn:
    # # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    # pass


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}#, 'min_samples_split':(18,20,22,24,26,28,30,32,34,36,38,40,42,44,46)}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # 1. Find an appropriate performance metric. This should be the same as the
    # one used in your performance_metric procedure above:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    metrics = {'median_ae' : m.median_absolute_error,  'mean_ae': m.mean_absolute_error,  'mse': m.mean_squared_error}

    # 2. We will use grid search to fine tune the Decision Tree Regressor and
    # obtain the parameters that generate the best training performance. Set up
    # the grid search object here.
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

    # for key in metrics:
    #     found_param = []
    #     for i in range(200):
    #         scorer = make_scorer(metrics[key], greater_is_better=False)
    #         reg = GridSearchCV(regressor, parameters, scoring=scorer)
    #         reg.fit(X,y)
    #         found_param.append(reg.best_params_['max_depth'])
    #     print key +"'s max_depth choice on average: ", np.mean(found_param), " with std: ", np.std(found_param)

    # The commented loop above resulted in these data:
    #
    # median_ae's max_depth choice on average:  5.685  with std:  0.696975609329
    # mean_ae's max_depth choice on average:  5.105  with std:  1.172166797
    # mse's max_depth choice on average:  5.425  with std:  1.73042624807
    
    # Which supports the observation that the error curves for mse and mean_ae were more erratic. Worse, running GridSearch with
    # mse would sometimes suggest an optimal max_depth as high as 8 or 9. Using median_ae, GridSearch would find a best max_depth
    # between 5 and 6 more consistently. Which is what we want, since it agrees with the behavior of the
    # Model_complexity_graph. Here, a clear divergence between the test and training error curves can be seen between max_depths
    # of 5 and 6. At that point the training error curve continues its descent towards zero (as it over fits) while the test error
    # curve levels off horizontally. So that after 5-6 max_depth the model would approach perfectly fitting the training data,
    # while not predicting unseen data with any better accuracy. I guess some of this would only matter if we were trying
    # automated the selection of max_depth. Similar behavior can be gleaned from the learning curve graphs. Basically for all
    # these curves, training error starts at zero and test error starts at a max. This is explained because few data points can
    # always be fit by a model, while the chances of that same model predicting an unseen example are low (heavily biased). As the
    # amount of data is incremented the curves approach each other since it is harder to fit all examples perfectly (training
    # set), but bias is being reduced (test set). For low values of max_depth this convergence levels off at a high error. This
    # indicates that the model is biased overall.  Neither the training set nor the test set are being fit particularly well. As
    # max_depth increases the horizontal where the two curves are converging, moves towards the x-axis. This is happening because
    # the more complex model is capable of fitting the training set better AND generalizing to the test set better. After a
    # max_depth of 5-6 however, only the training error horizontal continues this trend as it starts to over fit the data while no
    # better prediction is occurring with the test data. 

    scorer = make_scorer(metrics['median_ae'], greater_is_better=False)
    reg = GridSearchCV(regressor, parameters, scoring=scorer)

    print reg.fit(X,y).best_params_['max_depth']

    ### FINAL MODEL CHOICE: max_depth of 5 ###
    reg = DecisionTreeRegressor(max_depth = 5)
    # Fit the learner to the training data to obtain the best parameter set
    print "Final Model: "
    print reg.fit(X, y)    

    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = reg.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)

# In the case of the documentation page for GridSearchCV, it might be the case that the example is just a demonstration of syntax for use of the function, rather than a statement about 
def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    #Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()
