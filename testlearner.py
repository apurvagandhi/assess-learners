""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
	  	   		  		 		  		  		    	 		 		   		 		  
import math
import random  		  	   		  		 		  		  		    	 		 		   		 		  
import sys
import time
from matplotlib import pyplot as plt  		  	   		  		 		  		  		    	 		 		   		 		    	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  	  	   		  		 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  		
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import InsaneLearner as il  		  	   		  		 		  		  		    	 		 		   		 		   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		  		 		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		  		 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])  		  	   		  		 		  		  		    	 		 		   		 		  
    data = np.array([list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]  		  	   		  		 		  		  		    	 		 		   		 		  
    )  	

    # compute how much of the data is training and testing  
    random_row_indices = random.sample(range(data.shape[0]), data.shape[0])	
    number_of_train_rows = int(0.6 * data.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
    number_of_test_rows = data.shape[0] - number_of_train_rows 	  	   		  		 		  		  		    	 		 		   		 				 		   		 		  
    # separate out training and testing data  		 
    train_x = data[random_row_indices[:number_of_train_rows], 0:-1]  		  	   		  		 		  		  		    	 		 		   		 		  
    train_y = data[random_row_indices[:number_of_train_rows], -1]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_x = data[random_row_indices[number_of_test_rows:], 0:-1]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_y = data[random_row_indices[number_of_test_rows:], -1]  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		      		 		  		  		  	   		  		 		  		  		    	 		 		   		 		      

    # Experiment 1
    # initialize variables
    dt_insample_rmse_list = []
    dt_outsample_rmse_list = []
    dt_insample_mae_list = []
    dt_outsample_mae_list = []
    rt_insample_mae_list = []
    rt_outsample_mae_list = []
    bag_learner_insample_rmse_list = []
    bag_learner_outsample_rmse_list = []
    dt_time_list = []
    rt_time_list = []
    dt_time_list_query = []
    rt_time_list_query = []

    max_leafsize = 100
    
    # Experiment 1
    # Run DTLearner with different leaf_sizes
    for leafsize in range(max_leafsize):
        start_time = time.time()
        dt_learner = dtl.DTLearner(leafsize, verbose=True)
        dt_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        in_sample_dt_learner_pred_y = dt_learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - in_sample_dt_learner_pred_y) ** 2).sum() / train_y.shape[0])
        dt_insample_rmse_list.append(rmse)

        # evaluate out of sample rmse & std
        out_sampel_dt_learner_pred_y = dt_learner.query(test_x)
        rmse = math.sqrt(((test_y - out_sampel_dt_learner_pred_y) ** 2).sum() / test_y.shape[0])
        dt_outsample_rmse_list.append(rmse)

    # Plot rmse against leaf_size for experiment 1
    plt.plot(dt_insample_rmse_list)
    plt.plot(dt_outsample_rmse_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend(["In Sample RMSE", "Out Sample RMSE"])
    plt.title('DTLearner RMSE using different Leaf Sizes')
    plt.grid(True)
    plt.savefig('Experiment1.png')
    plt.clf()
    
    # Experiment 2 
    # Run BagLearner usuing DTLearner with different leaf_sizes and fixed bags
    for leafsize in range(max_leafsize):
        bag_learner = bl.BagLearner(learner = dtl.DTLearner, kwargs = {"leaf_size":leafsize}, bags = 20, boost = False, verbose = False)
        bag_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        in_sample_bag_learner_pred_y = bag_learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - in_sample_bag_learner_pred_y) ** 2).sum() / train_y.shape[0])
        bag_learner_insample_rmse_list.append(rmse)

        # evaluate out of sample rmse
        out_sample_bag_learner_pred_y = bag_learner.query(test_x)
        rmse = math.sqrt(((test_y - out_sample_bag_learner_pred_y) ** 2).sum() / test_y.shape[0])
        bag_learner_outsample_rmse_list.append(rmse)

    # Plot rmse against leaf_size for experiment 2
    plt.plot(bag_learner_insample_rmse_list)
    plt.plot(bag_learner_outsample_rmse_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend(["In Sample RMSE", "Out Sample RMSE"])
    plt.grid(True)
    plt.title('BagLearner using DTLearner RMSE using different Leaf Sizes and 10 bags')
    plt.savefig('Experiment2.png')
    plt.clf()	  	   	
           
    # Experiment 3
    # Run DTLearner with different leaf_sizes
    for leafsize in range(max_leafsize):
        start_time = time.time()
        dt_learner = dtl.DTLearner(leafsize, verbose=True)
        dt_learner.add_evidence(train_x, train_y)
        end_time = time.time()
        time_taken = end_time - start_time
        dt_time_list.append(time_taken)

        # evaluate in sample
        dt_learner_in_sample_pred_y = dt_learner.query(train_x)  # get the predictions
        
        # evaluate out of sample rmse & std
        start_time = time.time()
        dt_leaner_out_sample_pred_y = dt_learner.query(test_x)
        end_time = time.time()
        time_taken = end_time - start_time
        dt_time_list_query.append(time_taken)

        # Calculate the Mean Absolute Error (MAE) by taking the average of absolute errors
        for actual, predicted in zip(train_x, dt_learner_in_sample_pred_y):
            dt_learner_in_sample_absolute_errors = abs(actual - predicted) 
        dt_insample_mae_list.append(sum(dt_learner_in_sample_absolute_errors) / len(dt_learner_in_sample_absolute_errors))
            
        for actual, predicted in zip(test_x, dt_leaner_out_sample_pred_y):
            dt_learner_out_sample_absolute_errors = abs(actual - predicted) 
        dt_outsample_mae_list.append(sum(dt_learner_out_sample_absolute_errors) / len(dt_learner_out_sample_absolute_errors))
            
    # Run RTLearner with different leaf_sizes
    for leafsize in range(max_leafsize):
        start_time = time.time()
        rt_learner = rtl.RTLearner(leafsize, verbose=True)
        rt_learner.add_evidence(train_x, train_y)
        end_time = time.time()
        time_taken = end_time - start_time
        rt_time_list.append(time_taken)

        # evaluate in sample
        rt_learner_in_sample_pred_y = rt_learner.query(train_x)  # get the predictions
        
        # evaluate out of sample rmse & std
        start_time = time.time()
        rt_learner_out_sample_pred_y = rt_learner.query(test_x)
        end_time = time.time()
        time_taken = end_time - start_time
        rt_time_list_query.append(time_taken)
                
        # Calculate the Mean Absolute Error (MAE) by taking the average of absolute errors
        for actual, predicted in zip(train_x, rt_learner_in_sample_pred_y):
            rt_learner_in_sample_absolute_errors = abs(actual - predicted) 
        rt_insample_mae_list.append(sum(rt_learner_in_sample_absolute_errors) / len(rt_learner_in_sample_absolute_errors))
            
        for actual, predicted in zip(test_x, rt_learner_out_sample_pred_y):
            rt_learner_out_sample_absolute_errors = abs(actual - predicted) 
        rt_outsample_mae_list.append(sum(rt_learner_out_sample_absolute_errors) / len(rt_learner_out_sample_absolute_errors))
        
    plt.plot(dt_time_list)
    plt.plot(rt_time_list)
    plt.plot(dt_time_list_query)
    plt.plot(rt_time_list_query)

    plt.xlabel('Leaf Size')
    plt.ylabel('Time')
    plt.legend(["Train DT Tree Learner", "Train RT Learner","Query DT Tree Learner","Query RT Tree Learner"])
    plt.grid(True)
    plt.title('Decision Tree vs Random Tree Learners Run Time Comparision for Training')
    plt.savefig('Experiment3.1.png')
    plt.clf()
    
    plt.plot(rt_insample_mae_list)
    plt.plot(dt_insample_mae_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error')
    plt.legend(["Random Tree Learner", "Decision Tree Learner"])
    plt.grid(True)
    plt.title('Mean Absolute Error using In-Sample')
    plt.savefig('Experiment3.2.png')
    plt.clf()
    
    plt.plot(rt_outsample_mae_list)
    plt.plot(dt_outsample_mae_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error')
    plt.legend(["Random Tree Learner", "Decision Tree Learner"])
    plt.grid(True)
    plt.title('Mean Absolute Error using Out-Sample')
    plt.savefig('Experiment3.3.png')
    plt.clf()