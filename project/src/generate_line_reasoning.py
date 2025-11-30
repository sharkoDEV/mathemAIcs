import json
import os
import random
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

CANVAS = (256, 256)
GRID_COLOR = (210, 210, 210)
AXIS_COLOR = (150, 150, 150)
LINE_COLORS = [(52, 152, 219), (231, 76, 60)]
FONT = ImageFont.load_default()


def draw_grid(draw: ImageDraw.ImageDraw) -> None:
    step = 32
    for x in range(0, CANVAS[0], step):
        draw.line((x, 0, x, CANVAS[1]), fill=GRID_COLOR)
    for y in range(0, CANVAS[1], step):
        draw.line((0, y, CANVAS[0], y), fill=GRID_COLOR)
    cx = CANVAS[0] // 2
    cy = CANVAS[1] // 2
    draw.line((0, cy, CANVAS[0], cy), fill=AXIS_COLOR, width=2)
    draw.line((cx, 0, cx, CANVAS[1]), fill=AXIS_COLOR, width=2)


def line_points(slope: float, intercept: float) -> List[Tuple[int, int]]:
    pts = []
    for x in range(-4, 5):
        y = slope * x + intercept
        px = CANVAS[0] // 2 + x * 32
        py = CANVAS[1] // 2 - y * 32
        pts.append((px, py))
    return pts


def reasoning_template(
    scenario: str,
    slope_values: Tuple[float, float],
    intercepts: Tuple[float, float],
) -> str:
    m1, m2 = slope_values
    b1, b2 = intercepts
    if scenario == "parallel":
        topic = "Lignes parallèles"
        explanation = "Deux droites sont parallèles lorsqu'elles admettent la même pente."
        calculations = (
            f"pente₁ = {m1:.2f} ; pente₂ = {m2:.2f} ; intercepts b₁ = {b1:.2f}, b₂ = {b2:.2f}.\n"
            "Comme pente₁ = pente₂, les droites gardent la même direction."
        )
        final = "Les droites sont parallèles car elles ont la même pente et ne se couperont jamais."
    else:
        topic = "Lignes perpendiculaires"
        explanation = (
            "Dans un repère orthonormé, deux droites sont perpendiculaires si le produit de leurs pentes vaut -1."
        )
        calculations = (
            f"pente₁ = {m1:.2f} ; pente₂ = {m2:.2f} ; produit = {m1:.2f} × {m2:.2f} = {m1*m2:.2f}.\n"
            "Le produit vaut -1, donc les droites forment un angle droit."
        )
        final = "Les droites sont perpendiculaires car leurs pentes sont inverses et opposées."
    reformulation = "Déterminer la relation géométrique entre les deux droites colorées."
    topic_line = f"Nous sommes dans le chapitre : {topic}."
    explanation_line = explanation + " On vérifie la condition avec les coefficients directeurs."
    final_answer = (
        f"1. Reformulation : {reformulation}\n"
        f"2. Identification : {topic_line}\n"
        f"3. Explication : {explanation_line}\n"
        f"4. Calculs détaillés : {calculations}\n"
        f"5. Réponse justifiée : {final}"
    )
    return final_answer


def generate_line_reasoning_sample(index: int, image_dir: str, annotation_dir: str) -> Dict:
    image = Image.new("RGB", CANVAS, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw_grid(draw)
    scenario = random.choice(["parallel", "perpendicular"])
    if scenario == "parallel":
        slope = random.choice([-2, -1, -0.5, 0.5, 1, 2])
        intercept1 = random.randint(-3, 3)
        intercept2 = intercept1 + random.choice([-3, -2, 2, 3])
        slopes = (slope, slope)
    else:
        slope = random.choice([-2, -1, -0.5, 0.5, 1, 2])
        slope2 = -1 / slope
        intercept1 = random.randint(-2, 2)
        intercept2 = random.randint(-2, 2)
        slopes = (slope, slope2)
    intercepts = (intercept1, intercept2)
    pts1 = line_points(slopes[0], intercept1)
    pts2 = line_points(slopes[1], intercept2)
    draw.line(pts1, fill=LINE_COLORS[0], width=4)
    draw.line(pts2, fill=LINE_COLORS[1], width=4)
    annotation = {
        "type": scenario,
        "slopes": slopes,
        "intercepts": intercepts,
    }
    answer = reasoning_template(scenario, slopes, intercepts)
    question = "Les deux droites sont-elles parallèles ou perpendiculaires ? Justifier."
    qas = [{"question": question, "answer": answer}]
    image_name = f"lines_{index:05d}.png"
    annotation_name = f"lines_{index:05d}.json"
    image_path = os.path.join(image_dir, image_name)
    annotation_path = os.path.join(annotation_dir, annotation_name)
    annotation_payload = {
        "image": image_name,
        "question": question,
        "answer": answer,
        "objects": [annotation],
        "qas": qas,
    }
    image.save(image_path)
    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(annotation_payload, f, indent=2)
    return {
        "image_path": image_path,
        "annotation_path": annotation_path,
        "qas": qas,
    }


def generate_line_reasoning_dataset(count: int, image_dir: str, annotation_dir: str) -> List[Dict]:
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    manifest: List[Dict] = []
    for idx in range(count):
        manifest.append(generate_line_reasoning_sample(idx, image_dir, annotation_dir))
    return manifest


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    images = os.path.join(dataset_dir, "images")
    annotations = os.path.join(dataset_dir, "annotations")
    records = generate_line_reasoning_dataset(5, images, annotations)
    print(f"Generated {len(records)} line reasoning samples.")
