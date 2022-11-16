import os
import torch

from read_audio_data import read_audio_regression

baseDir = "N:\\AUD_Data\\sampled\\"
#create a webdataset, in which the class attributes and labels are stored based on a common basename
#needs manual tar creation command to compress all of the data
def create_tar(path):
    print("creating new tar file...")


    files_found = []
    for root, dirs, files in os.walk(baseDir + path):
        for file in files:
            if not file.endswith(".wav"):
                print("classes have already been created, exiting...")
                continue
            files_found.append(file)

    print("files found: " + str(len(files_found)))


    
    count = 0

    for file in files_found:
        audio, coord = read_audio_regression(baseDir + path, file)
        # print("found loc is: "  + str(coord[0]) + " " + str(coord[1]) + " for file: " + file)
        if (count % (len(files_found)/100)) == 0:
            print("count has completed " + str(count % len(files_found)/100) + " at file " + str(count) )
        #TODO: fix collapsed printing of numbers, create full tensor to print
        torch.save(audio, baseDir + "tars\\" + path + "\\" + file[:-4] + ".pt")

        torch.save(coord, baseDir + "tars\\" + path + "\\" + file[:-4] + ".txt")
        count +=1




# create_tar("train")
# create_tar("test")
create_tar("validation")