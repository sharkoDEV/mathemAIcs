import json
import os
import random
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from utils.multilingual import multilingual_prompt

CANVAS = (256, 256)
BACKGROUND = (248, 248, 248)
INK_COLORS = [
    (20, 20, 20),
    (40, 40, 80),
    (70, 20, 20),
    (10, 90, 10),
]
TEXT_SNIPPETS = [
    "x + y = 12",
    "P(A) = 0.35",
    "sin(theta)=0.5",
    "a^2+b^2=c^2",
    "f(x)=3x-4",
    "Delta ABC",
    "slope=2",
    "radius 5",
    "Median=7",
    "Area=24",
    "Angle 60 deg",
    "Thales",
    "Vector u",
    "Segment PQ",
    "Point (3,4)",
    "Line y=x",
    "8 cm",
    "12 m",
    "Base 4",
]


def random_arithmetic_snippet() -> str:
    terms = random.randint(2, 4)
    current = random.randint(1, 12)
    expression = [str(current)]
    value = current
    for _ in range(terms - 1):
        op = random.choice(["+", "-", "*"])
        operand = random.randint(1, 12)
        expression.append(f" {op} {operand}")
        if op == "+":
            value += operand
        elif op == "-":
            value -= operand
        else:
            value *= operand
    if random.random() < 0.25:
        divisor = random.randint(1, 12)
        value *= divisor
        expression = [f"{value}", " / ", f"{divisor}"]
        value //= divisor
    return "".join(expression) + f" = {value}"


def collect_fonts(limit: int = 25) -> List[str]:
    candidates = font_manager.findSystemFonts(fontext="ttf")
    random.shuffle(candidates)
    selected = []
    for path in candidates:
        if len(selected) >= limit:
            break
        selected.append(path)
    if not selected:
        selected.append(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")))
    return selected


FONT_FILES = collect_fonts()


def draw_text_blocks(draw: ImageDraw.ImageDraw, count: int) -> Tuple[List[Dict], List[Dict[str, str]]]:
    annotations: List[Dict] = []
    qas: List[Dict[str, str]] = []
    used_boxes: List[Tuple[int, int, int, int]] = []
    for idx in range(count):
        if random.random() < 0.7:
            snippet = random_arithmetic_snippet()
        else:
            snippet = random.choice(TEXT_SNIPPETS)
        font_path = random.choice(FONT_FILES)
        font_size = random.randint(14, 28)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox_measure = draw.textbbox((0, 0), snippet, font=font)
        w = bbox_measure[2] - bbox_measure[0]
        h = bbox_measure[3] - bbox_measure[1]
        if w == 0 or h == 0:
            continue
        max_x = CANVAS[0] - w - 10
        max_y = CANVAS[1] - h - 10
        if max_x < 5 or max_y < 5:
            continue
        x = random.randint(5, max_x)
        y = random.randint(5, max_y)
        bbox = draw.textbbox((x, y), snippet, font=font)
        overlap = any(
            not (bbox[2] < obox[0] or bbox[0] > obox[2] or bbox[3] < obox[1] or bbox[1] > obox[3])
            for obox in used_boxes
        )
        if overlap:
            continue
        color = random.choice(INK_COLORS)
        draw.text((x, y), snippet, font=font, fill=color)
        used_boxes.append(bbox)
        question = multilingual_prompt("point", coordinate=f"box {idx+1}") + " Identify the text inside."
        annotations.append({
            "text": snippet,
            "bbox": bbox,
            "font": os.path.basename(font_path),
            "color": color,
        })
        qas.append({
            "question": question,
            "answer": snippet,
        })
    return annotations, qas


def generate_text_panel(index: int, image_dir: str, annotation_dir: str) -> Dict:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    block_count = random.randint(2, 5)
    text_boxes, qas = draw_text_blocks(draw, block_count)
    image_name = f"textpanel_{index:05d}.png"
    annotation_name = f"textpanel_{index:05d}.json"
    image_path = os.path.join(image_dir, image_name)
    annotation_path = os.path.join(annotation_dir, annotation_name)
    annotation = {
        "image": image_name,
        "text_boxes": text_boxes,
        "qas": qas,
        "question": qas[0]["question"] if qas else "",
        "answer": qas[0]["answer"] if qas else "",
        "objects": [{"type": "text_box", "content": box["text"], "bbox": box["bbox"]} for box in text_boxes],
    }
    image.save(image_path)
    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
    return {
        "image_path": image_path,
        "annotation_path": annotation_path,
        "qas": qas,
    }


def generate_text_panel_dataset(count: int, image_dir: str, annotation_dir: str) -> List[Dict]:
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    records: List[Dict] = []
    for idx in range(count):
        records.append(generate_text_panel(idx, image_dir, annotation_dir))
    return records


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    images = os.path.join(dataset_dir, "images")
    annotations = os.path.join(dataset_dir, "annotations")
    records = generate_text_panel_dataset(5, images, annotations)
    print(f"Generated {len(records)} text panels.")
