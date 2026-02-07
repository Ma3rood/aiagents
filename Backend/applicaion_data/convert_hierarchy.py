#!/usr/bin/env python3
"""Convert ma3rood.csv (id, parent_id, name, parent_name) to hierarchy lines: parent, child, child..."""

import csv
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "ma3rood.csv"
OUTPUT_FILE = Path(__file__).parent / "ma3rood_hierarchy.csv"


def main():
    # id -> {name, parent_id}
    nodes = {}
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = row.get("id", "").strip()
            pid = row.get("parent_id", "").strip()
            name = row.get("name", "").strip()
            if nid and name:
                nodes[nid] = {"name": name, "parent_id": pid if pid and pid.upper() != "NULL" else None}

    def path_to_root(nid):
        id_path = []
        name_path = []
        current = nid
        while current:
            node = nodes.get(current)
            if not node:
                break
            id_path.append(current)
            name_path.append(node["name"])
            current = node["parent_id"]
        id_path.reverse()
        name_path.reverse()
        return id_path, name_path

    # ids that are someone's parent
    parent_ids = {nodes[n]["parent_id"] for n in nodes if nodes[n]["parent_id"]}

    # leaf nodes = have no children
    leaf_ids = [n for n in nodes if n not in parent_ids]

    rows = []
    for nid in leaf_ids:
        id_path, name_path = path_to_root(nid)
        if id_path:
            rows.append((id_path, name_path))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_path", "category_path"])
        for id_path, name_path in rows:
            writer.writerow([" > ".join(id_path), " > ".join(name_path)])

    print(f"Wrote {len(rows)} hierarchy lines to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
