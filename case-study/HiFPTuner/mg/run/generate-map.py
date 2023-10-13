import json

map_dict = {}
origconfig = json.loads(open("config.json", 'r').read())
for i in origconfig:
    func = origconfig[i]["function"]
    name = origconfig[i]["name"]
    key = f"{func}.{name}"
    map_dict[key] = i

with open('map.json', 'w', encoding='utf-8') as f:
    json.dump(map_dict, f, ensure_ascii=False, indent=4)
    print("map.json is generated.")