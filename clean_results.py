import glob
import os

# remove generated graphs
fileList = glob.glob("./classification_results/learning_data_*.png")
for filePath in fileList:
    try:
        os.remove(filePath)
        #print(filePath)
    except:
        print("Error while deleting file : ", filePath)

# remove the graphs_filtered directory
path = "./classification_results"
try:
    os.rmdir(path)
    #print("success on dir %s" % path)
except OSError:
    print ("Deletion of the directory %s failed" % path)

print("clean: successful")