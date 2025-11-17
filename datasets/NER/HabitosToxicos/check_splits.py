import os
import argparse
from collections import Counter, defaultdict
import statistics
import glob

def load_brat_dir(path):
    """
    Carga los archivos BRAT (texto + anotaciones).
    Devuelve:
    - num_docs
    - longitudes de los textos
    - distribución de entidades (Counter)
    """
    txt_files = sorted(glob.glob(os.path.join(path, "*.txt")))
    ann_files = [f.replace(".txt", ".ann") for f in txt_files]

    lengths = []
    entity_counter = Counter()

    for txt_f, ann_f in zip(txt_files, ann_files):
        # Longitud del texto
        with open(txt_f, "r", encoding="utf8") as f:
            text = f.read().strip()
            lengths.append(len(text))

        # Entidades
        if os.path.exists(ann_f):
            with open(ann_f, "r", encoding="utf8") as f:
                for line in f:
                    if line.startswith("T"):  # entidad normal
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            etype = parts[1].split()[0]
                            entity_counter[etype] += 1

    return {
        "num_docs": len(txt_files),
        "lengths": lengths,
        "entities": entity_counter,
    }


def print_split_stats(name, stats):
    print(f"\n===== {name.upper()} =====")
    print(f"Nº documentos: {stats['num_docs']}")

    if stats["lengths"]:
        print(f"Longitud media: {statistics.mean(stats['lengths']):.1f}")
        print(f"Longitud mediana: {statistics.median(stats['lengths']):.1f}")

    print("\nDistribución de entidades:")
    total = sum(stats["entities"].values())
    if total == 0:
        print("   (No se encontraron entidades)")
    else:
        for ent, count in stats["entities"].most_common():
            pct = 100 * count / total
            print(f"   {ent:20s} {count:5d} ({pct:5.1f}%)")


def compare_splits(stats_train, stats_dev, stats_test):
    print("\n\n================ COMPARACIÓN ENTRE SPLITS ================\n")

    # Compara distribución total de entidades
    sets = {
        "train": stats_train,
        "dev": stats_dev,
        "test": stats_test,
    }

    # Entidades globales
    all_entities = set()
    for v in sets.values():
        all_entities |= set(v["entities"].keys())

    print("DISTRIBUCIÓN RELATIVA POR ENTIDAD:")
    print("(Porcentaje sobre el total de entidades en cada split)\n")

    hdr = f"{'Entidad':20s} | {'train %':>8} | {'dev %':>8} | {'test %':>8}"
    print(hdr)
    print("-" * len(hdr))

    for ent in sorted(all_entities):
        row = [ent]
        for split_name, split_stats in sets.items():
            total = sum(split_stats["entities"].values())
            pct = 100 * split_stats["entities"].get(ent, 0) / total if total > 0 else 0
            row.append(f"{pct:8.2f}")
        print(f"{row[0]:20s} | {row[1]} | {row[2]} | {row[3]}")

    print("\nSi ves porcentajes muy diferentes entre dev y test → DISTRIBUTION SHIFT.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Ruta a la carpeta que contiene train/, dev/, test/")
    args = parser.parse_args()

    splits = ["train", "dev", "test"]
    stats = {}

    for split in splits:
        split_path = os.path.join(args.data_dir, split, "brat")
        if not os.path.exists(split_path):
            raise ValueError(f"No existe: {split_path}")
        stats[split] = load_brat_dir(split_path)

    # Imprimir stats individuales
    for split in splits:
        print_split_stats(split, stats[split])

    # Comparar splits
    compare_splits(stats["train"], stats["dev"], stats["test"])


if __name__ == "__main__":
    main()
