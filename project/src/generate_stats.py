import json
import random
from typing import Dict, List

import numpy as np
from sympy import symbols, integrate

from utils.multilingual import multilingual_prompt


def stats_problem() -> Dict:
    size = random.randint(4, 8)
    numbers = [random.randint(1, 20) for _ in range(size)]
    mean_value = float(np.mean(numbers))
    median_value = float(np.median(numbers))
    variance_value = float(np.var(numbers))
    prompt = (
        "Given the numbers "
        + ", ".join(map(str, numbers))
        + ", compute the mean, median, and variance and explain what each measure represents."
    )
    answer = (
        f"mean={mean_value:.2f}, median={median_value:.2f}, variance={variance_value:.2f}; "
        "mean balances all values, median locates the central value, variance describes spread."
    )
    return {"prompt": prompt, "answer": answer}


def integral_problem() -> Dict:
    x = symbols("x")
    coeff = random.randint(1, 5)
    exponent = random.randint(1, 3)
    expr = coeff * x ** exponent
    result = integrate(expr, (x, 0, 1))
    prompt = multilingual_prompt("integral", expr=f"{coeff}*x^{exponent}")
    answer = f"{float(result):.4f}"
    return {"prompt": prompt, "answer": answer}


def probability_problem() -> Dict:
    sides = random.choice([4, 6, 8])
    favorable = random.randint(1, sides)
    prompt = (
        f"A fair {sides}-sided die is rolled. What is the probability of rolling a value ≤ {favorable}? "
        "Explain why the probability equals favorable outcomes divided by the total outcomes."
    )
    prob = favorable / sides
    answer = (
        f"{prob:.2f}; there are {favorable} favorable results among {sides} equally likely outcomes, "
        "so probability=favorable/total."
    )
    return {"prompt": prompt, "answer": answer}


def arithmetic_problem() -> Dict:
    length = random.randint(3, 5)
    operations = ["+", "-", "*"]
    terms = []
    for i in range(length):
        term = random.randint(-20, 20)
        if i == 0:
            terms.append(str(term))
        else:
            op = random.choice(operations)
            terms.append(f" {op} {term}")
    expression = "".join(terms)
    result = eval(expression)
    prompt = expression
    answer = str(result)
    return {"prompt": prompt, "answer": answer}


def linear_equation_problem() -> Dict:
    a = random.randint(2, 8)
    b = random.randint(-10, 10)
    c = random.randint(-10, 10)
    solution = (c - b) / a
    prompt = (
        f"Solve the first-degree equation {a}x + {b} = {c}. Explain how to verify the solution."
    )
    answer = (
        f"x={solution:.2f}. Substitute: {a}*{solution:.2f}+{b}={a*solution+b:.2f}={c}, "
        "so the equality holds and the value is a solution."
    )
    return {"prompt": prompt, "answer": answer}


def proportion_percentage_problem() -> Dict:
    base = random.randint(20, 120)
    percent = random.choice([5, 10, 12.5, 15, 20, 25])
    prompt = (
        f"A quantity equals {base}. Explain how to identify a proportional situation and compute the result "
        f"after a {percent}% increase."
    )
    new_value = base * (1 + percent / 100)
    answer = (
        f"Ratios stay constant when the situation is proportional. Use coefficient 1+{percent}/100 to get "
        f"{new_value:.2f}. Percent change multiplies the base."
    )
    return {"prompt": prompt, "answer": answer}


def coordinate_repere_problem() -> Dict:
    x = random.randint(-5, 5)
    y = random.randint(-5, 5)
    prompt = (
        f"In a repère orthonormé, describe how to place the point ({x},{y}) and explain why orthonormal axes matter."
    )
    answer = (
        f"From origin move {x} along x then {y} along y; orthonormal axes guarantee unit lengths and right angles, "
        "making coordinates and distances reliable."
    )
    return {"prompt": prompt, "answer": answer}


def point_distance_problem() -> Dict:
    x1, y1 = random.randint(-5, 5), random.randint(-5, 5)
    x2, y2 = random.randint(-5, 5), random.randint(-5, 5)
    dx = x2 - x1
    dy = y2 - y1
    dist = (dx ** 2 + dy ** 2) ** 0.5
    prompt = (
        f"Compute the distance between A({x1},{y1}) and B({x2},{y2}) and explain why the Pythagorean theorem appears."
    )
    answer = (
        f"d=√(({dx})^2+({dy})^2)={dist:.2f}. The segment forms the hypotenuse of a right triangle with legs |{dx}| and |{dy}|, "
        "so Pythagoras applies."
    )
    return {"prompt": prompt, "answer": answer}


def pythagoras_problem() -> Dict:
    a = random.randint(3, 6)
    b = random.randint(4, 7)
    c = (a ** 2 + b ** 2) ** 0.5
    prompt = (
        f"Explain how to use the Pythagorean theorem to justify that a triangle with legs {a} and {b} is right "
        "and how to apply the converse."
    )
    answer = (
        f"If the third side squared equals {a**2 + b**2}, the triangle is right. "
        f"Here c=√({a}^2+{b}^2)≈{c:.2f}. Conversely, checking c^2={a**2 + b**2} proves a right angle."
    )
    return {"prompt": prompt, "answer": answer}


def affine_function_problem() -> Dict:
    a = random.choice([-3, -2, -1, 0.5, 1, 2, 3])
    b = random.randint(-5, 5)
    prompt = (
        f"Consider y={a}x+{b}. Explain how to recognize an affine function, interpret a and b, "
        "and why its graph is a straight line."
    )
    answer = (
        f"a={a} is the slope (variation of y per unit x), b={b} is the intercept (value at x=0). "
        "Linear dependence on x makes the graph a straight line."
    )
    return {"prompt": prompt, "answer": answer}


def graph_reading_problem() -> Dict:
    slope = random.choice([0.5, 1, 2])
    intercept = random.randint(-3, 3)
    x_val = random.randint(-2, 4)
    y_val = slope * x_val + intercept
    prompt = (
        f"A line is drawn on a graph. Explain how to read the value at x={x_val}, interpret the slope, "
        "and why a graph encodes a relationship."
    )
    answer = (
        f"Move vertically from x={x_val} to the line to read y={y_val:.2f}. "
        f"Slope {slope} tells how y changes for each unit x. A graph represents all ordered pairs satisfying the relation."
    )
    return {"prompt": prompt, "answer": answer}


def average_median_problem() -> Dict:
    data = [random.randint(1, 20) for _ in range(7)]
    data.sort()
    mean_value = sum(data) / len(data)
    median_value = data[len(data) // 2]
    prompt = f"Given data {data}, compute the mean and median and explain why they describe different information."
    answer = (
        f"mean={mean_value:.2f}, median={median_value}. Mean balances all data; median reports the central observation, "
        "so they can differ when data are skewed."
    )
    return {"prompt": prompt, "answer": answer}


def simple_probability_explanation() -> Dict:
    total = random.randint(3, 8)
    favorable = random.randint(1, total)
    prompt = (
        f"In an experiment with {total} equally likely outcomes, how do you compute the probability of an event "
        f"with {favorable} favorable outcomes and interpret the result?"
    )
    prob = favorable / total
    answer = (
        f"P={prob:.2f}= {favorable}/{total}. Probability counts favorable cases divided by total equally likely cases; "
        "it measures how often we expect the event in repeated trials."
    )
    return {"prompt": prompt, "answer": answer}


def parallel_lines_problem() -> Dict:
    slope = random.choice([-2, -1, 0.5, 1, 2])
    b1 = random.randint(-4, 4)
    b2 = b1 + random.randint(1, 5)
    prompt = (
        f"Explain how to prove the lines y={slope}x+{b1} and y={slope}x+{b2} are parallel, "
        "why equal slopes imply parallelism, and how corresponding angles appear."
    )
    answer = (
        f"Equal slopes {slope} give identical direction vectors so the lines never meet. "
        "Transversals cut parallel lines producing equal corresponding angles."
    )
    return {"prompt": prompt, "answer": answer}


def perpendicular_lines_problem() -> Dict:
    a = random.choice([1, 2, -1, -2])
    prompt = (
        f"Show that lines with slopes {a} and {-1/a} are perpendicular in an orthonormal repère and "
        "describe how to recognize a right angle on a diagram."
    )
    answer = (
        f"Product of slopes {a}*{-1/a}=-1 proves perpendicularity. Right angles are noted by a square marker or 90° label."
    )
    return {"prompt": prompt, "answer": answer}


PROBLEM_CREATORS = [
    stats_problem,
    integral_problem,
    probability_problem,
    arithmetic_problem,
    linear_equation_problem,
    proportion_percentage_problem,
    coordinate_repere_problem,
    point_distance_problem,
    pythagoras_problem,
    affine_function_problem,
    graph_reading_problem,
    average_median_problem,
    simple_probability_explanation,
    parallel_lines_problem,
    perpendicular_lines_problem,
]


def generate_stat_text_dataset(count: int) -> List[Dict]:
    samples: List[Dict] = []
    for _ in range(count):
        creator = random.choice(PROBLEM_CREATORS)
        samples.append(creator())
    return samples


def write_jsonl(records: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    data = generate_stat_text_dataset(10)
    write_jsonl(data, "text_samples.jsonl")
