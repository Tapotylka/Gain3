"""Чтение файлов формата TSPLIB (EUC_2D) — стандартной библиотеки тестовых задач коммивояжёра."""
from pathlib import Path

from tspga.instance import TSPInstance

# Известные оптимумы задач TSPLIB (длины в целочисленной метрике EUC_2D)
KNOWN_OPTIMA = {
    "eil51": 426, "berlin52": 7542, "st70": 675, "eil76": 538, "kroA100": 21282, "kroB100": 22141,
    "kroC100": 20749, "kroD100": 21294, "kroE100": 22068, "rd100": 7910, "eil101": 629, "ch130": 6110,
    "ch150": 6528, "kroA150": 26524, "kroA200": 29368, "a280": 2579,
}


def parse_tsplib(text: str, default_name: str = "tsplib") -> TSPInstance:
    header, coords, in_coords = {}, [], False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "EOF":
            break
        if line == "NODE_COORD_SECTION":
            in_coords = True
        elif in_coords:
            _, x, y = line.split()[:3]
            coords.append((float(x), float(y)))
        elif ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().upper()] = value.strip()
    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError(f"поддерживается только EDGE_WEIGHT_TYPE: EUC_2D (в файле: {header.get('EDGE_WEIGHT_TYPE')})")
    if int(header.get("DIMENSION", -1)) != len(coords):
        raise ValueError("DIMENSION не совпадает с числом строк NODE_COORD_SECTION")
    name = header.get("NAME", default_name)
    return TSPInstance.from_coords(name, coords, rounded=True, known_optimum=KNOWN_OPTIMA.get(name))


def load_tsplib(path) -> TSPInstance:
    path = Path(path)
    return parse_tsplib(path.read_text(encoding="utf-8"), default_name=path.stem)
