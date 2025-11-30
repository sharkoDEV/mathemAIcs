import json
import math
import os
import random
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils.multilingual import multilingual_prompt

CANVAS_SIZE = (256, 256)
FONT = ImageFont.load_default()
BACKGROUND = (255, 255, 255)
FOREGROUND = (30, 30, 30)
GRID_COLOR = (180, 180, 180)


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def polygon_area(points: List[Tuple[int, int]]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def random_color() -> Tuple[int, int, int]:
    return tuple(random.randint(50, 220) for _ in range(3))


def draw_grid(draw: ImageDraw.ImageDraw) -> None:
    step = 32
    for x in range(0, CANVAS_SIZE[0], step):
        draw.line((x, 0, x, CANVAS_SIZE[1]), fill=GRID_COLOR, width=1)
    for y in range(0, CANVAS_SIZE[1], step):
        draw.line((0, y, CANVAS_SIZE[0], y), fill=GRID_COLOR, width=1)


def draw_axes(draw: ImageDraw.ImageDraw) -> None:
    cx = CANVAS_SIZE[0] // 2
    cy = CANVAS_SIZE[1] // 2
    draw.line((0, cy, CANVAS_SIZE[0], cy), fill=FOREGROUND, width=2)
    draw.line((cx, 0, cx, CANVAS_SIZE[1]), fill=FOREGROUND, width=2)
    for offset in range(-4, 5):
        if offset == 0:
            continue
        x = cx + offset * 32
        y = cy + offset * 32
        draw.line((x, cy - 5, x, cy + 5), fill=FOREGROUND, width=1)
        draw.line((cx - 5, y, cx + 5, y), fill=FOREGROUND, width=1)
        draw.text((x - 4, cy + 8), str(offset), fill=FOREGROUND, font=FONT)
        draw.text((cx + 8, y - 4), str(-offset), fill=FOREGROUND, font=FONT)


def scenario_triangle(draw: ImageDraw.ImageDraw, annotation: Dict) -> List[Dict[str, str]]:
    points = [
        (random.randint(40, 210), random.randint(40, 210))
        for _ in range(3)
    ]
    draw.polygon(points, outline=FOREGROUND, fill=random_color())
    perim = sum(distance(points[i], points[(i + 1) % 3]) for i in range(3))
    area = polygon_area(points)
    annotation["objects"].append({
        "type": "triangle",
        "points": points,
        "area": round(area, 2),
        "perimeter": round(perim, 2)
    })
    point_text = ", ".join(str(tuple(p)) for p in points)
    qas = [
        {
            "question": multilingual_prompt("triangle", points=point_text),
            "answer": f"area={area:.2f}, perimeter={perim:.2f}",
        },
        {
            "question": multilingual_prompt("triangle", points=point_text) + " Provide only the area.",
            "answer": f"area={area:.2f}",
        },
        {
            "question": multilingual_prompt("triangle", points=point_text) + " Provide only the perimeter.",
            "answer": f"perimeter={perim:.2f}",
        },
    ]
    return qas


def scenario_rectangle(draw: ImageDraw.ImageDraw, annotation: Dict) -> List[Dict[str, str]]:
    x1, y1 = random.randint(20, 120), random.randint(20, 120)
    width = random.randint(40, 100)
    height = random.randint(40, 100)
    x2, y2 = x1 + width, y1 + height
    draw.rectangle((x1, y1, x2, y2), outline=FOREGROUND, width=3)
    area = width * height
    perim = 2 * (width + height)
    diagonal = distance((x1, y1), (x2, y2))
    annotation["objects"].append({
        "type": "rectangle",
        "coordinates": [x1, y1, x2, y2],
        "area": area,
        "perimeter": perim,
        "width": width,
        "height": height,
        "diagonal": round(diagonal, 2),
    })
    question = multilingual_prompt("rectangle", width=width, height=height)
    qas = [
        {
            "question": question,
            "answer": f"area={area}, perimeter={perim}",
        },
        {
            "question": question + " Focus on the area value.",
            "answer": f"area={area}",
        },
        {
            "question": question + " Give the diagonal length.",
            "answer": f"diagonal={diagonal:.2f}",
        },
    ]
    return qas


def scenario_circle(draw: ImageDraw.ImageDraw, annotation: Dict) -> List[Dict[str, str]]:
    radius = random.randint(20, 60)
    center = (random.randint(radius + 20, CANVAS_SIZE[0] - radius - 20),
              random.randint(radius + 20, CANVAS_SIZE[1] - radius - 20))
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    draw.ellipse(bbox, outline=FOREGROUND, width=3)
    area = math.pi * radius ** 2
    circumference = 2 * math.pi * radius
    diameter = 2 * radius
    annotation["objects"].append({
        "type": "circle",
        "center": center,
        "radius": radius,
        "area": round(area, 2),
        "circumference": round(circumference, 2),
        "diameter": diameter,
    })
    question = multilingual_prompt("circle", radius=radius)
    qas = [
        {
            "question": question,
            "answer": f"area={area:.2f}, circumference={circumference:.2f}",
        },
        {
            "question": question + " Provide only the area.",
            "answer": f"area={area:.2f}",
        },
        {
            "question": question + " Provide only the diameter.",
            "answer": f"diameter={diameter}",
        },
    ]
    return qas


def scenario_segment(draw: ImageDraw.ImageDraw, annotation: Dict) -> List[Dict[str, str]]:
    p1 = (random.randint(30, 220), random.randint(30, 220))
    p2 = (random.randint(30, 220), random.randint(30, 220))
    draw.line((*p1, *p2), fill=FOREGROUND, width=3)
    draw.text((p1[0] + 5, p1[1] + 5), "A", fill=FOREGROUND, font=FONT)
    draw.text((p2[0] + 5, p2[1] + 5), "B", fill=FOREGROUND, font=FONT)
    length = distance(p1, p2)
    midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    annotation["objects"].append({
        "type": "segment",
        "points": [p1, p2],
        "length": round(length, 2),
        "midpoint": tuple(round(v, 2) for v in midpoint),
        "length_squared": round(length ** 2, 2),
    })
    base = multilingual_prompt("segment", p1=str(p1), p2=str(p2))
    qas = [
        {"question": base, "answer": f"length={length:.2f}"},
        {"question": base + " Provide the squared length.", "answer": f"length_squared={length ** 2:.2f}"},
        {"question": base + " Provide the midpoint.", "answer": f"midpoint=({midpoint[0]:.2f},{midpoint[1]:.2f})"},
    ]
    return qas


def scenario_polygon(draw: ImageDraw.ImageDraw, annotation: Dict) -> List[Dict[str, str]]:
    vertex_count = random.randint(4, 6)
    points = [(random.randint(40, 210), random.randint(40, 210)) for _ in range(vertex_count)]
    draw.line(points + [points[0]], fill=FOREGROUND, width=2)
    area = polygon_area(points)
    perim = sum(distance(points[i], points[(i + 1) % vertex_count]) for i in range(vertex_count))
    annotation["objects"].append({
        "type": "polygon",
        "points": points,
        "vertices": vertex_count,
        "area": round(area, 2),
        "perimeter": round(perim, 2)
    })
    point_text = ", ".join(str(tuple(p)) for p in points)
    question = multilingual_prompt("polygon", points=point_text)
    qas = [
        {"question": question, "answer": f"area={area:.2f}, perimeter={perim:.2f}"},
        {"question": question + " Respond only with the area.", "answer": f"area={area:.2f}"},
        {"question": question + " Respond only with the number of vertices.", "answer": f"vertices={vertex_count}"},
    ]
    return qas


def scenario_labeled_point(draw: ImageDraw.ImageDraw, annotation: Dict) -> List[Dict[str, str]]:
    draw_grid(draw)
    draw_axes(draw)
    gx = random.randint(-4, 4)
    gy = random.randint(-4, 4)
    pixel = (CANVAS_SIZE[0] // 2 + gx * 32, CANVAS_SIZE[1] // 2 - gy * 32)
    draw.ellipse((pixel[0] - 5, pixel[1] - 5, pixel[0] + 5, pixel[1] + 5), fill="red")
    label = f"P({gx},{gy})"
    draw.text((pixel[0] + 6, pixel[1] - 16), label, fill="red", font=FONT)
    annotation["objects"].append({
        "type": "point",
        "coordinate": [gx, gy],
        "label": label
    })
    question = multilingual_prompt("point", coordinate=str((gx, gy)))
    qas = [
        {"question": question, "answer": label},
        {"question": question + " Respond only with the x value.", "answer": f"x={gx}"},
        {"question": question + " Respond only with the y value.", "answer": f"y={gy}"},
    ]
    return qas


def scenario_line(draw: ImageDraw.ImageDraw, annotation: Dict) -> List[Dict[str, str]]:
    draw_grid(draw)
    draw_axes(draw)
    slope = random.choice([-2, -1, -0.5, 0.5, 1, 2])
    intercept = random.randint(-2, 2)
    points = []
    for x in range(-4, 5):
        y = slope * x + intercept
        px = CANVAS_SIZE[0] // 2 + x * 32
        py = CANVAS_SIZE[1] // 2 - y * 32
        points.append((px, py))
    draw.line(points, fill="blue", width=3)
    annotation["objects"].append({
        "type": "line",
        "equation": f"y={slope}x+{intercept}",
        "slope": slope,
        "intercept": intercept,
        "value_at_1": round(slope * 1 + intercept, 2),
    })
    base_question = multilingual_prompt("line", equation=f"y={slope}x+{intercept}")
    qas = [
        {"question": base_question, "answer": f"slope={slope}, intercept={intercept}"},
        {"question": base_question + " Provide only the slope.", "answer": f"slope={slope}"},
        {"question": base_question + " Provide the intercept value.", "answer": f"intercept={intercept}"},
        {"question": base_question + " Evaluate y when x=1.", "answer": f"y={slope * 1 + intercept:.2f}"},
    ]
    return qas


SCENARIOS = [
    scenario_triangle,
    scenario_rectangle,
    scenario_circle,
    scenario_segment,
    scenario_polygon,
    scenario_labeled_point,
    scenario_line,
]


def generate_geometry_sample(index: int, image_dir: str, annotation_dir: str) -> Dict:
    image = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    annotation: Dict = {"objects": []}
    scenario = random.choice(SCENARIOS)
    qas = scenario(draw, annotation)
    image_name = f"geometry_{index:05d}.png"
    annotation_name = f"geometry_{index:05d}.json"
    image_path = os.path.join(image_dir, image_name)
    annotation_path = os.path.join(annotation_dir, annotation_name)
    annotation.update({
        "image": image_name,
        "question": qas[0]["question"],
        "answer": qas[0]["answer"],
        "qas": qas,
    })
    image.save(image_path)
    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
    return {
        "image_path": image_path,
        "annotation_path": annotation_path,
        "qas": qas,
    }


def generate_geometry_dataset(count: int, image_dir: str, annotation_dir: str) -> List[Dict]:
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    manifest: List[Dict] = []
    for idx in range(count):
        manifest.append(generate_geometry_sample(idx, image_dir, annotation_dir))
    return manifest


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    images = os.path.join(dataset_dir, "images")
    annotations = os.path.join(dataset_dir, "annotations")
    records = generate_geometry_dataset(10, images, annotations)
    print(f"Generated {len(records)} geometry samples.")
