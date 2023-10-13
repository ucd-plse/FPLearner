import json


include = {}
include["lbm.c"] = {}
include["lbm.c"]["function"] = {}
# add variables to exclude for each function to include
include["lbm.c"]["function"]["LBM_performStreamCollideBGK"] = [""]
include["lbm.c"]["function"]["LBM_performStreamCollideTRT"] = [""]
include["lbm.c"]["function"]["LBM_handleInOutFlow"] = [""]
include["lbm.c"]["function"]["LBM_showGridStatistics"] = [""]
include["lbm.c"]["function"]["LBM_compareVelocityField"] = ["epsilon", "error"]



with open('include.json', 'w', encoding='utf-8') as f:
    json.dump(include, f, ensure_ascii=False, indent=4)
    print("include.json is generated.")

