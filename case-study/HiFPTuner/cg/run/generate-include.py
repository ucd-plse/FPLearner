import json


include = {}
include["cg.c"] = {}
include["cg.c"]["function"] = {}
# add variables to exclude for each function to include
include["cg.c"]["function"]["main"] = ["t", "mflops", "tmax", "zeta_verify_value", "epsilon", "err"]
include["cg.c"]["function"]["makea"] = ["aelt"]
include["cg.c"]["function"]["conj_grad"] = [""]
include["cg.c"]["function"]["sprnvc"] = [""]
include["cg.c"]["function"]["vecset"] = [""]
include["cg.c"]["function"]["sparse"] = ["aelt"]
include["cg.c"]["function"]["icnvrt"] = [""]


with open('include.json', 'w', encoding='utf-8') as f:
    json.dump(include, f, ensure_ascii=False, indent=4)
    print("include.json is generated.")

