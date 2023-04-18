import glob 


for i in glob.glob("/data/disk1/hungpham/object-detection-generator/images/trash_4d/*"):
    name = i.split("/")[-1]
    print(name)
    with open("infor_trash.txt", "a") as f:
        f.write("logo,images/trash_4d/{},0 \n".format(name))