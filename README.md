1) Code that trains a model as well as the extra infrastructure required to extract the structed correlation data from images of graphs





#### Old info disregard ####
To generate graphs:
Call execute_function.py with "create_training_data" and update instructions.yaml accordingly to give the desired instructions. Alternitively, you can call with another argument which is a different .yaml file containing instructions on how to generate graphs.

To clean the directories of any old graphs:
Call execute_function.py with "clean"
Notes: this removes any graphs in any of the directories used for training or validation
and removes the .csv file containing information about the graphs.

To train a model that can classify graphs:
Call execute_function.py with "graph_classification" and update instructions.yaml accordingly to give the desired instructions. Alternitively, you can call with another argument which is a different .yaml file containing instructions on how to classify graphs.

To train a model that can classify how many series are present in a graph (this is required to determine the k for k-means when segmenting out the different color series):
Call execute_function.py with "series_classification" and update instructions.yaml accordingly to give the desired instructions. Alternitively, you can call with another argument which is a different .yaml file containing instructions on how to classify graphs.

To clean the directory and files regarding saved classification results:
Call clean_resutls.py

To clean the directory and files regarding segmented images:
Call execute_function.py with "cleanSeries"