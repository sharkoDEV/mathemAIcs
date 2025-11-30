import argparse
import json
import os
import random
from typing import Dict, List

from generate_charts import generate_chart_dataset
from generate_line_reasoning import generate_line_reasoning_dataset
from generate_shapes import generate_geometry_dataset
from generate_stats import generate_stat_text_dataset, write_jsonl
from generate_text_panels import generate_text_panel_dataset
from utils.multilingual import multilingual_prompt

BLANK_PROMPT_PROB = 0.2

def annotation_to_text(annotation_path: str) -> List[Dict[str, str]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qas = data.get("qas")
    if qas:
        return [{"prompt": qa["question"], "answer": qa["answer"]} for qa in qas if qa.get("question") and qa.get("answer")]
    objects = data.get("objects", [])
    if not objects:
        question = data.get("question")
        answer = data.get("answer")
        if question and answer:
            return [{"prompt": question, "answer": answer}]
        return []
    primary = objects[0]
    obj_type = primary.get("type")
    if obj_type == "triangle":
        prompt = multilingual_prompt(
            "triangle", points=", ".join(str(tuple(p)) for p in primary["points"])
        )
    elif obj_type == "rectangle":
        prompt = multilingual_prompt(
            "rectangle", width=primary["width"], height=primary["height"]
        )
    elif obj_type == "circle":
        prompt = multilingual_prompt("circle", radius=primary["radius"])
    elif obj_type == "segment":
        prompt = multilingual_prompt(
            "segment",
            p1=str(tuple(primary["points"][0])),
            p2=str(tuple(primary["points"][1])),
        )
    elif obj_type == "point":
        prompt = multilingual_prompt("point", coordinate=str(tuple(primary["coordinate"])))
    elif obj_type == "line":
        prompt = multilingual_prompt("line", equation=primary["equation"])
    elif obj_type == "polygon":
        prompt = multilingual_prompt(
            "polygon", points=", ".join(str(tuple(p)) for p in primary["points"])
        )
    else:
        prompt = data.get("question", "Describe the object.")
    return [{"prompt": prompt, "answer": data.get("answer", "")}]


def build_dataset(geometry: int, charts: int, text_panels: int, line_reasoning: int, text_samples: int, base_dir: str) -> None:
    dataset_dir = os.path.join(base_dir, "dataset")
    image_dir = os.path.join(dataset_dir, "images")
    annotation_dir = os.path.join(dataset_dir, "annotations")
    text_dir = os.path.join(dataset_dir, "text")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    geometry_manifest = generate_geometry_dataset(geometry, image_dir, annotation_dir)
    chart_manifest = generate_chart_dataset(charts, image_dir, annotation_dir)
    text_panel_manifest = generate_text_panel_dataset(text_panels, image_dir, annotation_dir)
    line_reason_manifest = generate_line_reasoning_dataset(line_reasoning, image_dir, annotation_dir)
    combined_manifest = geometry_manifest + chart_manifest + text_panel_manifest + line_reason_manifest

    manifest_path = os.path.join(annotation_dir, "dataset_manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in combined_manifest:
            rel_image = os.path.relpath(record["image_path"], base_dir)
            rel_annotation = os.path.relpath(record["annotation_path"], base_dir)
            for qa in record.get("qas", []):
                prompt = qa["question"]
                if random.random() < BLANK_PROMPT_PROB:
                    prompt = ""
                f.write(
                    json.dumps(
                        {
                            "image": rel_image,
                            "annotation": rel_annotation,
                            "prompt": prompt,
                            "answer": qa["answer"],
                            "has_image": True,
                        }
                    )
                    + "\n"
                )

    stats_records = generate_stat_text_dataset(text_samples)
    for record in combined_manifest:
        extra = annotation_to_text(record["annotation_path"])
        if extra:
            stats_records.extend(extra)

    text_path = os.path.join(text_dir, "math_text.jsonl")
    write_jsonl(stats_records, text_path)
    print(
        f"Saved {len(combined_manifest)} multimodal samples and {len(stats_records)} text samples."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build synthetic math datasets")
    parser.add_argument("--geometry", type=int, default=50)
    parser.add_argument("--charts", type=int, default=50)
    parser.add_argument("--ocr", type=int, default=50, help="Number of text panel images")
    parser.add_argument("--lines", type=int, default=40, help="Number of line reasoning images")
    parser.add_argument("--text", type=int, default=200)
    args = parser.parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    build_dataset(args.geometry, args.charts, args.ocr, args.lines, args.text, project_root)
