"""Pre-registered whole-curve consensus shuffle for the P1-A control."""
import hashlib
import json
import os
from pathlib import Path
import random


def shuffle_profiles(profiles, grad_cache):
    seed = int(os.environ["SPECCON_SHUFFLE_SEED"])
    output = Path(os.environ["SPECCON_SHUFFLE_MANIFEST"])
    groups = {}
    gradient_hash = hashlib.sha256()
    for name in sorted(profiles):
        gradient = grad_cache[name]["grad"].detach().cpu().contiguous()
        shape = list(gradient.shape)
        key = (name.rsplit(".", 1)[-1], tuple(shape))
        groups.setdefault(key, []).append(name)
        gradient_hash.update(name.encode())
        gradient_hash.update(gradient.numpy().tobytes())
    rng = random.Random(seed)
    mapping = {}
    for key in sorted(groups):
        names = sorted(groups[key])
        if len(names) == 1:
            mapping[names[0]] = names[0]
        else:
            rng.shuffle(names)
            mapping.update({name: names[(i + 1) % len(names)] for i, name in enumerate(names)})
    source_hash = hashlib.sha256()
    for name in sorted(profiles):
        source_hash.update(name.encode())
        source_hash.update(profiles[name]["consistency"].numpy().tobytes())
    shuffled = {}
    for name, profile in profiles.items():
        shuffled[name] = dict(profile)
        shuffled[name]["consistency"] = profiles[mapping[name]]["consistency"].clone()
    payload = {
        "control": "whole_curve_consensus_derangement_within_leaf_and_matrix_shape",
        "seed": seed, "mapping": mapping,
        "groups": [{"module_type": key[0], "shape": list(key[1]), "count": len(names)}
                   for key, names in sorted(groups.items())],
        "unchanged_modules": sum(name == donor for name, donor in mapping.items()),
        "total_modules": len(mapping),
        "source_consensus_sha256": source_hash.hexdigest(),
        "aggregated_gradient_sha256": gradient_hash.hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return shuffled
