To generate graphs:
Call execute_function.py with "create_training_data" and any other flags and arguments you wish to use
Use -h for more info

To clean the directories of any old graphs:
Call execute_function.py with "clean"
Notes: this removes any graphs in any of the directories used for training or validation
and removes the .csv file containing information about the graphs.

To train a model that can classify graphs:
Call execute_function.py with "graph_classification"

To clean the directory and files regarding saved classification results
Call clean_resutls.py