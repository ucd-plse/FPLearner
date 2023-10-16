import json
import os
import sys


CONFIG=sys.argv[1]


with open('include.json') as json_file:
    include = json.load(json_file)

for file in include.keys():
    os.system("clang -Xclang -load -Xclang ../../plugin/TransformType.so -Xclang -plugin -Xclang trans-type -Xclang -plugin-arg-trans-type -Xclang -output-path -Xclang -plugin-arg-trans-type -Xclang ../tempscripts/ -Xclang -plugin-arg-trans-type -Xclang -input-config -Xclang -plugin-arg-trans-type -Xclang {} ../scripts/{}".format(CONFIG, file))