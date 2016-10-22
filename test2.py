from logistic_regression_template import *
import numpy as np

penalty = [0.001,0.01,0.1,1.0]

hyperparameters = {
                    'learning_rate': 0.03,
                    'weight_regularization': True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 450,
                    'weight_decay': 0.001 # related to standard deviation of weight prior 
                    }
for k in penalty:
	hyperparameters['weight_decay'] = k
	num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs
    plt.axis([0,hyperparameters['num_iterations'],0,200])
    train_ce = logging[:,1]
    valid_ce = logging[:,3]
    plt.plot(train_ce,color='g',ls='-',label="Training Cross Entropy")
    plt.plot(valid_ce,color='r',ls='-',label="Valid Cross Entropy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=5, borderaxespad=0.)
    plt.show()