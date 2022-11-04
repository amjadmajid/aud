import webdataset as wds
import os

from read_audio_data import loc_to_xy


#create a webdataset, in which the class attributes and labels are stored based on a common basename
#needs manual tar creation command to compress all of the data
def create_tar(path):
    print("creating new tar file...")


    files_found = []
    #TODO: define class labels
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".cls"):
                print("classes have already been created, exiting...")
                exit(-1)
            files_found.append(file)

    print("files found: " + str(len(files_found)))

    for file in files_found:
        x, y = loc_to_xy(file)
        print("found loc is: "  + str(x) + " " + str(y) + " for file: " + file)
        with open(path + "\\" + file[:-4] + ".cls", 'w') as fp:
            fp.write(str(x)+", " + str(y))
            pass




# create_tar("N:\\AUD_Data\\sampled\\tars\\train")
# create_tar("N:\\AUD_Data\\sampled\\tars\\test")
# create_tar("N:\\AUD_Data\\sampled\\tars\\validation")