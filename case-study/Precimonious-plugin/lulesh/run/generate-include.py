import json


include = {}
include["lulesh.cc"] = {}
include["lulesh.cc"]["function"] = {}
# add variables to exclude for each function to include
include["lulesh.cc"]["function"]["CalcCourantConstraintForElems"] = []
include["lulesh.cc"]["function"]["CalcHydroConstraintForElems"] = []
include["lulesh.cc"]["function"]["CalcElemVelocityGradient"] = ["b"]
include["lulesh.cc"]["function"]["CalcElemShapeFunctionDerivatives"] = ["x", "y", "z", "b", "volume"]
include["lulesh.cc"]["function"]["CalcElemCharacteristicLength"] = ["x", "y", "z", "volume"]
include["lulesh.cc"]["function"]["AreaFace"] = []

include["lulesh.cc"]["function"]["CalcVolumeForceForElems"] = ["sigxx", "sigyy", "sigzz", "determ"]
include["lulesh.cc"]["function"]["IntegrateStressForElems"] = ["sigxx", "sigyy", "sigzz", "determ", "fx_elem", "fy_elem", "fz_elem", "fx_local", "fy_local", "fz_local", "B", "x_local", "y_local", "z_local"]
include["lulesh.cc"]["function"]["SumElemFaceNormal"] = []
include["lulesh.cc"]["function"]["SumElemStressesToNodeForces"] = ["B", "fx", "fy", "fz"]
include["lulesh.cc"]["function"]["CalcHourglassControlForElems"] = ["determ", "dvdx", "dvdy", "dvdz", "x8n", "y8n", "z8n", "x1", "y1", "z1", "pfx", "pfy", "pfz"]
include["lulesh.cc"]["function"]["VoluDer"] = []
include["lulesh.cc"]["function"]["CalcFBHourglassForceForElems"] = ["determ", "x8n", "y8n", "z8n",  "dvdx", "dvdy", "dvdz", "fx_elem", "fy_elem", "fz_elem", "fx_local", "fy_local", "fz_local", "hgfx", "hgfy", "hgfz", "hourgam", "xd1", "yd1", "zd1"]    
include["lulesh.cc"]["function"]["CalcVelocityForNodes"] = []
include["lulesh.cc"]["function"]["CalcPositionForNodes"] = []
include["lulesh.cc"]["function"]["CalcKinematicsForElems"] = ["B", "D", "x_local", "y_local", "z_local", "xd_local", "yd_local", "zd_local"]
include["lulesh.cc"]["function"]["CalcElemVolume"] = []
include["lulesh.cc"]["function"]["CalcMonotonicQGradientsForElems"] = []
include["lulesh.cc"]["function"]["CalcMonotonicQRegionForElems"] = []
include["lulesh.cc"]["function"]["EvalEOSForElems"] = ["vnewc", "e_old", "delvc", "p_old", "q_old", "compression", "compHalfStep", "qq_old", "ql_old", "work", "p_new", "e_new", "q_new", "bvc", "pbvc"]
include["lulesh.cc"]["function"]["CalcEnergyForElems"] = ["pHalfStep", "p_new", "e_new", "q_new", "bvc", "pbvc", "p_old", "e_old", "q_old", "compression", "compHalfStep", "vnewc", "work", "delvc", "qq_old", "ql_old" ]
include["lulesh.cc"]["function"]["CalcPressureForElems"] = []
include["lulesh.cc"]["function"]["CalcSoundSpeedForElems"] = []
include["lulesh.cc"]["function"]["UpdateVolumesForElems"] = []




with open('include.json', 'w', encoding='utf-8') as f:
    json.dump(include, f, ensure_ascii=False, indent=4)
    print("include.json is generated.")
