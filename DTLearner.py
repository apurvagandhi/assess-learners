import numpy as np  

class DTLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size = 1, verbose=False):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   	
        self.leaf_size = leaf_size
        self.vrbose = verbose
           		  	   		  		 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        return "agandhi301"  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		  		 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		  		 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        
        data = np.column_stack((data_y, data_x))
        self.tree = self.build_tree(data)
    
    def build_tree(self, data):
        #If there is only one row
        data_x = data[:, 1:]
        data_y = data[:, 0]
        if data.shape[0] == 1 or data.shape[0] <= self.leaf_size:
            return np.array([['leaf', np.mean(data_y), None, None]])
        elif np.all(data_y == data_y[0]):
            return np.array([['leaf', data_y[0], None, None]])
        else: 
            # Calculate correlation coefficients between Y and each feature column
            correlations = np.abs(np.corrcoef(data_x, data_y, rowvar=False)[:-1, -1])
            # Find the index of the feature with the highest absolute correlation
            best_feature_index = np.argmax(correlations)
            # Get the name or label of the best feature
            splitVal = np.median(data[:, best_feature_index])
            left_tree = self.build_tree(data[data[:, best_feature_index] <= splitVal])
            right_tree = self.build_tree(data[data[:, best_feature_index] > splitVal])
            root = np.array([[best_feature_index, splitVal, 1, left_tree.shape[0] + 1]])
            return np.append(root, np.append(left_tree, right_tree, axis=0), axis=0)         
  		  	   		  		 		  		  		    	 		 		   		 		  
    def query(self, features):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	
        predicted_value = []
        for feature in features:
            row = 0
            node = self.tree[row,0]
            while (node != "leaf"): # if it is not a leaf node, enter loop
                splitVal = self.tree[row, 1]
                left = int(self.tree[row, 2])
                right = int(self.tree[row, 3])
                if(feature[int(node)] <= splitVal):
                    row = row + left
                else:
                    row = row + right
                node = self.tree[row,0]
            predicted_value.append(self.tree[row,1])
        return predicted_value
    	    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("Running DT Learner")  		  	   		  		 		  		  		    	 		 		   		 		  
