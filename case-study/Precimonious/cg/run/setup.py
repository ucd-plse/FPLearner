import os
import sys
# import json

BENCH=sys.argv[1]

os.system("rm -rf ../tempscripts/{}".format(BENCH.upper()))
os.system("rm -rf ../tempscripts/common")
os.system("cp -r ../scripts/{} ../tempscripts/".format(BENCH.upper()))
os.system("cp -r ../scripts/common ../tempscripts/")

os.system("make clean")
if not os.path.isdir("../tempscripts/bin"):
    os.system("mkdir ../tempscripts/bin")