import json


include = {}
include["funarc.c"] = {}
include["funarc.c"]["function"] = {}
# add variables to exclude for each function to include
include["funarc.c"]["function"]["main"] = ["epsilon", "zeta_verify_value"]
include["funarc.c"]["function"]["fun"] = [""]



with open('include.json', 'w', encoding='utf-8') as f:
    json.dump(include, f, ensure_ascii=False, indent=4)
    print("include.json is generated.")

