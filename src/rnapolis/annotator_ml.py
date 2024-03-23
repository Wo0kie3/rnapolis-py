#! /usr/bin/env python
import argparse
import csv
import logging
import os
import itertools
import numpy as np
from pandas import DataFrame
from xgboost import XGBClassifier
from typing import List, Optional, Tuple
import orjson
from scipy.spatial import KDTree

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    BpSeq,
    LeontisWesthof,
    Residue,
    Saenger,
    Stacking,
    Structure2D,
)
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import (
    AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT,
    Mapping2D3D,
    Residue3D,
    Structure3D,
)
from rnapolis.util import handle_input_file

C1P_MAX_DISTANCE = 10.0

LABELS = {
    "1": 1,
    "2": 2
}

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())


def _planar_angle(p1, p2, p3):
    b1 = p2 - p1
    b2 = p2 - p3

    angle = np.arccos(
        np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2)))
    return np.degrees(angle)


def _torsion_angle(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    torsion = np.arctan2(
        np.dot(np.cross(n1, n2), b2 * np.linalg.norm(b2)), np.dot(n1, n2)
    )
    return np.degrees(torsion)


def pairwise_distances(df: DataFrame, num_bins: int = 50) -> list:
    coords = df[['x', 'y', 'z']].values
    distances = np.sqrt(
        np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
    upper_triangle_indices = np.triu_indices_from(distances, k=1)
    flattened_distances = distances[upper_triangle_indices]
    histogram, bin_edges = np.histogram(
        flattened_distances, bins=num_bins, density=True)
    normalized_histogram = histogram / np.sum(histogram)
    return normalized_histogram, bin_edges


def plannar_angles(vector1: list, vector2: list, num_bins: int) -> list:

    pairs_from_vector1 = list(itertools.combinations(vector1, 2))
    pairs_from_vector2 = list(itertools.combinations(vector2, 2))

    # Generate triplets where two elements are from vector1 and one from vector2
    triplets_1 = []
    for single_element in vector2:
        for pair in pairs_from_vector1:
            triplets_1.append((*pair, single_element))

    # Generate triplets where two elements are from vector2 and one from vector1
    triplets_2 = []
    for single_element in vector1:
        for pair in pairs_from_vector2:
            triplets_2.append((*pair, single_element))

    # Combine all triplets
    all_triplets = triplets_1 + triplets_2

    angles = []

    all_triplets = np.array(all_triplets)

    angles = [_planar_angle(trip[0], trip[1], trip[2])
              for trip in all_triplets]

    histogram, _ = np.histogram(angles, bins=num_bins, density=True)
    normalized_histogram = histogram / np.sum(histogram)
    return normalized_histogram


def torsion_angles(vector_i: list, pos_i: int, vector_j: list, pos_j: int, num_bins: int) -> list:

    pairs_from_vector1 = [(vector_i[pos_i], atom)
                          for atom in vector_i if list(atom) != list(vector_i[pos_i])]

    pairs_from_vector2 = [(vector_j[pos_j], atom)
                          for atom in vector_j if list(atom) != list(vector_j[pos_j])]

    combinations_of_pairs = []
    for pair1 in pairs_from_vector1:
        for pair2 in pairs_from_vector2:
            combinations_of_pairs.append(pair1 + pair2)

    combinations_of_pairs = np.array(combinations_of_pairs)

    # Print the combinations of pairs
    angles = [_torsion_angle(quad[0], quad[1], quad[2], quad[3])
              for quad in combinations_of_pairs]

    histogram, bin_edges = np.histogram(angles, bins=num_bins, density=True)
    normalized_histogram = histogram / np.sum(histogram)
    return normalized_histogram, bin_edges


def is_correct_according_to_rnaview(residue: Residue3D):

    n1 = residue.find_atom('N1').coordinates
    c2 = residue.find_atom('C2').coordinates
    c6 = residue.find_atom('C6').coordinates

    d1 = np.linalg.norm(c2 - c6)
    d2 = np.linalg.norm(n1 - c6)
    d3 = np.linalg.norm(n1 - c2)

    return (d1 <= 3.0 and
            d2 <= 2.0 and
            d3 <= 2.0)


def residue_to_representation(residue_i: Residue3D, residue_j: Residue3D) -> list:

    atoms_i = [atom.coordinates for atom in residue_i.atoms]
    pos_i = [atom.name for atom in residue_i.atoms].index("C1'")
    atoms_j = [atom.coordinates for atom in residue_j.atoms]
    pos_j = [atom.name for atom in residue_j.atoms].index("C1'")

    distances = pairwise_distances(atoms_i, atoms_j)
    plannars = plannar_angles(atoms_i, atoms_j)
    torsions = plannar_angles(atoms_i, pos_i, atoms_j, pos_j)

    return [*distances, *plannars, *torsions]


def is_base_pair(residue_i: Residue3D, residue_j: Residue3D) -> bool:

    if not is_correct_according_to_rnaview(residue_i) and not is_correct_according_to_rnaview(residue_j):
        return False

    data = residue_to_representation(residue_i, residue_j)

    model = XGBClassifier()
    prediction = model.predict(data)

    return prediction


def classify_lw(residue_i: Residue3D, residue_j: Residue3D) -> Optional[LeontisWesthof]:

    data = residue_to_representation(residue_i, residue_j)

    model = XGBClassifier()
    prediction = model.predict(data)
    return prediction


def classify_saenger(residue_i: Residue3D, residue_j: Residue3D) -> Optional[Saenger]:
    data = residue_to_representation(residue_i, residue_j)

    model = XGBClassifier()
    prediction = model.predict(data)
    return prediction


def is_stacking(residue_i: Residue3D, residue_j: Residue3D) -> bool:
    data = residue_to_representation(residue_i, residue_j)

    model = XGBClassifier()
    prediction = model.predict(data)

    return prediction


def find_candidates(
    structure: Structure3D, model: Optional[int] = None
) -> List[Tuple[Residue3D, Residue3D]]:
    residue_map = {}
    coordinates = []

    for residue in structure.residues:
        if model is not None and residue.model != model:
            continue

        atom = residue.find_atom("C1'")

        if atom is not None:
            atom_xyz = (atom.x, atom.y, atom.z)
            residue_map[atom_xyz] = residue
            coordinates.append(atom_xyz)

    kdtree = KDTree(coordinates)
    candidates = []

    for i, j in kdtree.query_pairs(C1P_MAX_DISTANCE):
        residue_i = residue_map[coordinates[i]]
        residue_j = residue_map[coordinates[j]]
        candidates.append((residue_i, residue_j))

    return candidates


def find_pairs(structure: Structure3D, model: Optional[int] = None) -> List[BasePair]:
    base_pairs = []

    for residue_i, residue_j in find_candidates(structure, model):
        if is_base_pair(residue_i, residue_j):
            lw = classify_lw(residue_i, residue_j)
            saenger = classify_saenger(residue_i, residue_j)
            base_pairs.append(
                BasePair(
                    Residue(residue_i.label, residue_i.auth),
                    Residue(residue_j.label, residue_j.auth),
                    lw,
                    saenger,
                )
            )

    return base_pairs


def find_stackings(
    structure: Structure3D, model: Optional[int] = None
) -> List[Stacking]:
    stackings = []

    for residue_i, residue_j in find_candidates(structure, model):
        if is_stacking(residue_i, residue_j):
            stackings.append(
                Stacking(
                    Residue(residue_i.label, residue_i.auth),
                    Residue(residue_j.label, residue_j.auth),
                    None,
                )
            )

    return stackings


def extract_base_interactions(
    tertiary_structure: Structure3D, model: Optional[int] = None
) -> BaseInteractions:
    base_pairs = find_pairs(tertiary_structure, model)
    stackings = find_stackings(tertiary_structure, model)
    return BaseInteractions(base_pairs, stackings, [], [], [])


def extract_secondary_structure(
    tertiary_structure: Structure3D,
    model: Optional[int] = None,
    find_gaps: bool = False,
) -> BaseInteractions:
    base_interactions = extract_base_interactions(tertiary_structure, model)
    mapping = Mapping2D3D(
        tertiary_structure,
        base_interactions.basePairs,
        base_interactions.stackings,
        find_gaps,
    )
    stems, single_strands, hairpins, loops = mapping.bpseq.elements
    return Structure2D(
        base_interactions,
        str(mapping.bpseq),
        mapping.dot_bracket,
        mapping.extended_dot_bracket,
        stems,
        single_strands,
        hairpins,
        loops,
    )


def write_json(path: str, structure2d: BaseInteractions):
    with open(path, "wb") as f:
        f.write(orjson.dumps(structure2d))


def write_csv(path: str, structure2d: Structure2D):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["nt1", "nt2", "type", "classification-1", "classification-2"])
        for base_pair in structure2d.baseInteractions.basePairs:
            writer.writerow(
                [
                    base_pair.nt1.full_name,
                    base_pair.nt2.full_name,
                    "base pair",
                    base_pair.lw.value,
                    (
                        base_pair.saenger.value or ""
                        if base_pair.saenger is not None
                        else ""
                    ),
                ]
            )
        for stacking in structure2d.baseInteractions.stackings:
            writer.writerow(
                [
                    stacking.nt1.full_name,
                    stacking.nt2.full_name,
                    "stacking",
                    stacking.topology.value if stacking.topology is not None else "",
                    "",
                ]
            )
        for base_phosphate in structure2d.baseInteractions.basePhosphateInteractions:
            writer.writerow(
                [
                    base_phosphate.nt1.full_name,
                    base_phosphate.nt2.full_name,
                    "base-phosphate interaction",
                    base_phosphate.bph.value if base_phosphate.bph is not None else "",
                    "",
                ]
            )
        for base_ribose in structure2d.baseInteractions.basePhosphateInteractions:
            writer.writerow(
                [
                    base_ribose.nt1.full_name,
                    base_ribose.nt2.full_name,
                    "base-ribose interaction",
                    base_ribose.bph.value if base_ribose.bph is not None else "",
                    "",
                ]
            )
        for other in structure2d.baseInteractions.otherInteractions:
            writer.writerow(
                [
                    other.nt1.full_name,
                    other.nt2.full_name,
                    "other interaction",
                    "",
                    "",
                ]
            )


def write_bpseq(path: str, bpseq: BpSeq):
    with open(path, "w") as f:
        f.write(str(bpseq))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to PDB or mmCIF file")
    parser.add_argument("--bpseq", help="(optional) path to output BPSEQ file")
    parser.add_argument("--csv", help="(optional) path to output CSV file")
    parser.add_argument(
        "--json",
        help="(optional) path to output JSON file",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="(optional) if set, the program will print extended secondary structure to the standard output",
    )
    parser.add_argument(
        "--find-gaps",
        action="store_true",
        help="(optional) if set, the program will detect gaps and break the PDB chain into two or more strands; "
        f"the gap is defined as O3'-P distance greater then {
            1.5 * AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT}",
    )
    parser.add_argument("--dot", help="(optional) path to output DOT file")
    args = parser.parse_args()

    breakpoint()

    file = handle_input_file(args.input)
    structure3d = read_3d_structure(file, None)
    structure2d = extract_secondary_structure(
        structure3d, None, args.find_gaps)

    if args.csv:
        write_csv(args.csv, structure2d)

    if args.json:
        write_json(args.json, structure2d)

    if args.bpseq:
        write_bpseq(args.bpseq, structure2d.bpseq)

    if args.extended:
        print(structure2d.extendedDotBracket)
    else:
        print(structure2d.dotBracket)

    if args.dot:
        print(BpSeq.from_string(structure2d.bpseq).graphviz)


if __name__ == "__main__":
    main()
