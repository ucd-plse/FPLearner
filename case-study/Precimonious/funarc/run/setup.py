import os
import sys


file = 'funarc.c'

os.system("rm ../tempscripts/{}".format(file))
os.system("cp ../scripts/{} ../tempscripts/".format(file))
os.system("make clean")
