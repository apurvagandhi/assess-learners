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
    print(data)	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		  		 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		  		 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		  		 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		  		 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}") 
      		 		  		  		 
   	 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # create a learner and train it  		  	   		  		 		  		  		    	 		 		   		 		  
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		  		 		  		  		    	 		 		   		 		  
    learner.add_evidence(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
    print(learner.author())  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		  		 		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("In sample results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		  		 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		  		 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  	
    
    
    DTlearner = dtl.DTLearner(leaf_size = 1, verbose = False)  # create a DTLearner  		  	   		  		 		  		  		    	 		 		   		 		  
    DTlearner.add_evidence(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
    print(DTlearner.author())     
    pred_y_DT_Learner = DTlearner.query(train_x)  # get the predictions  		
    print("DT Learner",pred_y_DT_Learner)
    

        
    RTlearner = rtl.RTLearner(leaf_size = 1, verbose = False)  # create a DTLearner  		  	   		  		 		  		  		    	 		 		   		 		  
    RTlearner.add_evidence(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
    print(RTlearner.author())     
    pred_y_RT_Learner = RTlearner.query(train_x)  # get the predictions  	
    print("RT Learner", pred_y_RT_Learner)
    
    Baglearner = bl.BagLearner(learner = dtl.DTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)  # create a DTLearner  		  	   		  		 		  		  		    	 		 		   		 		  
    Baglearner.add_evidence(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
    print(Baglearner.author())     
    pred_y_Bag_Learner = Baglearner.query(train_x)  # get the predictions  	
    print("Bag Learner", pred_y_Bag_Learner)
    
    # #  -------------------------------- Create and Train DTlearner ---------------

    # # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner = dtl.DTLearner(1, verbose=True)  # DTLearner
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())

    # # # evaluate in sample
    # pred_y = learner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")

    # # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")

    # # ------- Experiment 1 and 3.1
    # # initialize variables
    # dt_insample_rmse_list = []
    # dt_outsample_rmse_list = []
    # dt_time_list = []
    # dt_insample_std_list = []
    # dt_outsample_std_list = []
    # max_leafsize = 50

    # # # Run DTLearner with different leaf_sizes
    # for leafsize in range(max_leafsize):
    #     start_time = time.time()
    #     learner = dtl.DTLearner(leafsize, verbose=True)
    #     learner.add_evidence(train_x, train_y)
    #     end_time = time.time()
    #     time_taken = end_time - start_time
    #     dt_time_list.append(time_taken)

    # #     # evaluate in sample
    #     pred_y = learner.query(train_x)  # get the predictions
    #     rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    #     dt_insample_rmse_list.append(rmse)
    #     dt_std = np.std(pred_y)
    #     dt_insample_std_list.append(dt_std)

    # #     # evaluate out of sample rmse & std
    #     pred_y = learner.query(test_x)
    #     rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    #     dt_outsample_rmse_list.append(rmse)
    #     dt_std = np.std(pred_y)
    #     dt_outsample_std_list.append(dt_std)

    # # # Plot rmse against leaf_size
    # # # insample_rmse_list.plot()
    # # # outsample_rmse_list.plot()
    # plt.plot(dt_insample_rmse_list)
    # plt.plot(dt_outsample_rmse_list)
    # plt.xlabel('Leaf Size')
    # plt.ylabel('RMSE')
    # plt.legend(["Train RMSE", "Test RMSE"])
    # plt.title('DTLearner RMSE using different Leaf Sizes')
    # plt.savefig('Experiment1.png')
    # plt.close()
	 		  	   		  		 		  		  		    	 		 		   		 		  
