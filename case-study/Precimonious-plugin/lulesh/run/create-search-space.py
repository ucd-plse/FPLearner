import json
import os


with open('include.json') as json_file:
    include = json.load(json_file)

for file in include.keys():
    os.system(f"clang -Xclang -load -Xclang ../../plugin/CreateSearchSpace.so -Xclang -plugin -Xclang create-space -Xclang -plugin-arg-create-space -Xclang -output-path -Xclang -plugin-arg-create-space -Xclang ./ -Xclang -plugin-arg-create-space -Xclang -output-name -Xclang -plugin-arg-create-space -Xclang config.json -Xclang -plugin-arg-create-space -Xclang -input-file -Xclang -plugin-arg-create-space -Xclang {file} ../scripts/{file}")

