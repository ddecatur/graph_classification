# script to clean out old graphs and directories
import os
import glob


def clean(directory):
    cwd=os.getcwd()
    if(cwd!=directory):
        print("error: clean called from wrong directory, could cause problems with deletions")
    else:
        # remove csv file
        try:
            os.remove("./graph_info.csv")
        except:
            print("Error while deleting graph_info.csv. Likely the file does not exist")

        # remove generated graphs
        fileList = glob.glob("./graphs_filtered/*/*/*graph*.png")
        for filePath in fileList:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)


        # remove correlation directories
        dirList = glob.glob("./graphs_filtered/*/*")
        for dirPath in dirList:
            dsPath = dirPath + "/.DS_Store"
            try:
                os.remove(dsPath)
                #print(dsPath)
                #print("success on .DS_Store removal from %s" % dirPath)
            except:
                print("Warning: error removing .DS_Store file, might not exist (this is ok)")
            try:
                os.rmdir(dirPath)
                #print("success on correlation dir %s" % dirPath)
            except OSError:
                print ("Deletion of the directory %s failed" % dirPath)

        # remove train and valadation directories
        dirList2 = glob.glob("./graphs_filtered/*")
        for dirPath2 in dirList2:
            dsPath2 = dirPath2 + "/.DS_Store"
            try:
                os.remove(dsPath2)
                #print(dsPath)
                #print("success on .DS_Store removal from %s" % dirPath2)
            except:
                print("Warning: error removing .DS_Store file, might not exist (this is ok)")
            try:
                os.rmdir(dirPath2)
                #print("success on dir %s" % dirPath2)
            except OSError:
                print ("Deletion of the directory %s failed" % dirPath2)

        # remove the graphs_filtered directory
        path = "./graphs_filtered"
        dsPath3 = path + "/.DS_Store"
        try:
            os.remove(dsPath3)
            #print(dsPath2)
            #print("success on .DS_Store removal from %s" % path)
        except:
            print("Warning: error removing .DS_Store file, might not exist (this is ok)")
        try:
            os.rmdir(path)
            #print("success on dir %s" % path)
        except OSError:
            print ("Deletion of the directory %s failed" % path)

        print("clean: successful")

def cleanReg(directory):
    cwd=os.getcwd()
    if(cwd!=directory):
        print("error: cleanReg called from wrong directory, could cause problems with deletions")
    else:
        # remove generated graphs
        fileList = glob.glob("./graphs_filtered/*/*/reg_*graph*.png")
        for filePath in fileList:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)