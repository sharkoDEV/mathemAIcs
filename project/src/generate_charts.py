import json
import os
import random
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.multilingual import multilingual_prompt

plt.style.use("seaborn-v0_8")


def scatter_question(ax) -> Tuple[Dict, List[Dict[str, str]]]:
    count = random.randint(20, 40)
    data = np.random.randn(count, 2) * random.uniform(0.5, 1.5)
    ax.scatter(data[:, 0], data[:, 1], c="tab:blue")
    means = data.mean(axis=0)
    annotation = {
        "type": "scatter",
        "points": data.tolist(),
        "mean": means.tolist()
    }
    ax.set_title("Random Scatter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    qas = [
        {
            "question": multilingual_prompt("scatter", detail="the mean of all X coordinates rounded to two decimals"),
            "answer": f"mean_x={means[0]:.2f}",
        },
        {
            "question": multilingual_prompt("scatter", detail="the mean of all Y coordinates rounded to two decimals"),
            "answer": f"mean_y={means[1]:.2f}",
        },
        {
            "question": multilingual_prompt("scatter", detail="how many samples appear in the scatter plot"),
            "answer": f"count={count}",
        },
    ]
    return annotation, qas


def line_question(ax) -> Tuple[Dict, List[Dict[str, str]]]:
    slope = random.choice([-2, -1, -0.5, 0.5, 1, 2])
    intercept = random.uniform(-1, 1)
    x = np.linspace(-3, 3, 100)
    y = slope * x + intercept
    noise = np.random.randn(len(x)) * 0.1
    y_noisy = y + noise
    ax.plot(x, y_noisy, label="line", color="tab:orange")
    ax.grid(True)
    ax.set_title("Line Plot")
    annotation = {
        "type": "line_plot",
        "slope": slope,
        "intercept": round(intercept, 2)
    }
    qas = [
        {"question": multilingual_prompt("line_plot", detail="the slope of the noisy line"), "answer": f"slope={slope}"},
        {
            "question": multilingual_prompt("line_plot", detail="the intercept of the noisy line"),
            "answer": f"intercept={intercept:.2f}",
        },
        {
            "question": multilingual_prompt("line_plot", detail="the value of y when x equals 1"),
            "answer": f"y={slope * 1 + intercept:.2f}",
        },
    ]
    return annotation, qas


def histogram_question(ax) -> Tuple[Dict, List[Dict[str, str]]]:
    data = np.random.randint(0, 10, size=200)
    bins = np.arange(-0.5, 10.5, 1)
    counts, edges, _ = ax.hist(data, bins=bins, color="tab:green")
    ax.set_title("Histogram")
    max_bin = int(np.argmax(counts))
    annotation = {
        "type": "histogram",
        "counts": counts.astype(int).tolist(),
        "bins": edges.tolist(),
        "max_bin_index": max_bin
    }
    total = int(counts.sum())
    qas = [
        {"question": multilingual_prompt("histogram", detail="which integer bin occurs most frequently"), "answer": f"mode={max_bin}"},
        {
            "question": multilingual_prompt("histogram", detail="how many samples were used in total"),
            "answer": f"count={total}",
        },
        {
            "question": multilingual_prompt("histogram", detail="how many entries fall inside the modal bin"),
            "answer": f"max_bin_count={int(counts[max_bin])}",
        },
    ]
    return annotation, qas


def bar_chart_question(ax) -> Tuple[Dict, List[Dict[str, str]]]:
    categories = ["A", "B", "C", "D", "E"]
    values = [random.randint(2, 15) for _ in categories]
    ax.bar(categories, values, color="tab:purple")
    ax.set_title("Bar Chart")
    max_idx = int(np.argmax(values))
    min_idx = int(np.argmin(values))
    annotation = {
        "type": "bar_chart",
        "categories": categories,
        "values": values,
        "max_category": categories[max_idx],
        "min_category": categories[min_idx],
    }
    qas = [
        {"question": multilingual_prompt("bar_chart", detail="which category has the highest bar"), "answer": categories[max_idx]},
        {"question": multilingual_prompt("bar_chart", detail="which category has the lowest bar"), "answer": categories[min_idx]},
        {
            "question": multilingual_prompt("bar_chart", detail="what is the difference between the tallest and shortest bars"),
            "answer": f"diff={values[max_idx] - values[min_idx]}",
        },
    ]
    return annotation, qas


CHART_CREATORS = [
    scatter_question,
    line_question,
    histogram_question,
    bar_chart_question,
]


def generate_chart_sample(index: int, image_dir: str, annotation_dir: str) -> Dict:
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    creator = random.choice(CHART_CREATORS)
    chart_data, qas = creator(ax)
    image_name = f"chart_{index:05d}.png"
    annotation_name = f"chart_{index:05d}.json"
    image_path = os.path.join(image_dir, image_name)
    annotation_path = os.path.join(annotation_dir, annotation_name)
    fig.tight_layout()
    fig.savefig(image_path)
    plt.close(fig)
    annotation = {
        "image": image_name,
        "question": qas[0]["question"],
        "answer": qas[0]["answer"],
        "chart": chart_data,
        "qas": qas,
    }
    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
    return {
        "image_path": image_path,
        "annotation_path": annotation_path,
        "qas": qas,
    }


def generate_chart_dataset(count: int, image_dir: str, annotation_dir: str) -> List[Dict]:
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    manifest: List[Dict] = []
    for idx in range(count):
        manifest.append(generate_chart_sample(idx, image_dir, annotation_dir))
    return manifest


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    images = os.path.join(dataset_dir, "images")
    annotations = os.path.join(dataset_dir, "annotations")
    records = generate_chart_dataset(10, images, annotations)
    print(f"Generated {len(records)} chart samples.")
