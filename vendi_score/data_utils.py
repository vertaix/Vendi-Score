from dataclasses import asdict, dataclass, field
import random
from typing import Any, Dict, List, Optional


@dataclass
class Example:
    x: Any
    features: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Group:
    name: str = ""
    examples: List[Example] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    Ks: Dict[str, Any] = field(default_factory=dict)
    ws: Dict[str, Any] = field(default_factory=dict)
    vs: Dict[str, Any] = field(default_factory=dict)


def to_batches(lst, batch_size):
    batches = []
    i = 0
    while i < len(lst):
        batches.append(lst[i : i + batch_size])
        i += batch_size
    return batches


def mode_dropping_groups(label_to_examples, categories, N):
    lst = []
    for i in range(1, len(categories) + 1):
        Ns = [N // i for _ in range(i)]
        while sum(Ns) < N:
            j = random.choice(range(len(Ns)))
            Ns[j] += 1
        group = Group(i, [])
        for n, cat in zip(Ns, categories):
            group.examples += random.sample(label_to_examples[cat], n)
        lst.append(group)
    return lst
