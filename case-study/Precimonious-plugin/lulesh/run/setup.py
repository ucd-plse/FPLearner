import os
import sys


file = 'lulesh.cc'

os.system("rm ../tempscripts/{}".format(file))
os.system("cp ../scripts/{} ../tempscripts/".format(file))
os.system("make clean")
