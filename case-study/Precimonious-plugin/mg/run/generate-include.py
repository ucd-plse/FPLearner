import json


include = {}
include["mg.c"] = {}
include["mg.c"]["function"] = {}
# add variables to exclude for each function to include
include["mg.c"]["function"]["main"] = ["mflops", "t", "tinit", "epsilon", "verify_value", "err", "tmax"]
include["mg.c"]["function"]["zero3"] = [""]
include["mg.c"]["function"]["zran3"] = ["x", "ten"]
include["mg.c"]["function"]["norm2u3"] = ["rnmu"]
include["mg.c"]["function"]["mg3P"] = [""]
include["mg.c"]["function"]["power"] = [""]
include["mg.c"]["function"]["bubble"] = ["ten"]
include["mg.c"]["function"]["comm3"] = [""]
include["mg.c"]["function"]["rep_nrm"] = [""]
include["mg.c"]["function"]["showall"] = [""]
include["mg.c"]["function"]["rprj3"] = [""]
include["mg.c"]["function"]["psinv"] = [""]
include["mg.c"]["function"]["interp"] = [""]


with open('include.json', 'w', encoding='utf-8') as f:
    json.dump(include, f, ensure_ascii=False, indent=4)
    print("include.json is generated.")
