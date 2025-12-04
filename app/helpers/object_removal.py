import json

def get_object_center(data_json): 
    data = json.loads(data_json)
    points = data["points"]
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    img_w = data["image"]["width"]
    img_h = data["image"]["height"]

    x = center_x * img_w
    y = center_y * img_h

    return x, y