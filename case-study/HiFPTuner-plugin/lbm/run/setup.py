import os


os.system("rm ../tempscripts/*")
os.system("cp ../scripts/* ../tempscripts/")

os.system("make clean")