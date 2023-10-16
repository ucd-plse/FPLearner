import os
import sys
# import json

BENCH=sys.argv[1]

os.system("rm -r ../tempscripts/{}".format(BENCH.upper()))
os.system("rm -r ../tempscripts/common")
os.system("cp -r ../scripts/{} ../tempscripts/".format(BENCH.upper()))
os.system("cp -r ../scripts/common ../tempscripts/")

os.system("make clean")
os.system("mkdir ../tempscripts/bin")