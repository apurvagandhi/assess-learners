import numpy as np  
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False):  
        self.BagLearners = []
        for i in range(20):
            self.BagLearners.append(bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False))
    def author(self):  		  	   		  		 		  		  		    	 		 		   		 		  
        return "agandhi301"  		  	   		  		 		  		  		    	 		 		   		 		  	  		 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  	
        for learner in self.BagLearners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):  		  	   		  		 		  		  		    	 		 		   		 		  
        for leaner in self.BagLearners:
            y_pred = leaner.query(points)
        return np.mean(y_pred, axis = 0)
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("Running insane learner")