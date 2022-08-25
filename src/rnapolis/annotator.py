#! /usr/bin/env python
import argparse
import math
from collections import Counter, defaultdict
from typing import IO, Dict, List, Optional, Tuple

import numpy
import numpy.typing
import orjson
from scipy.spatial import KDTree

from rnapolis.common import (
    BR,
    BasePair,
    BasePhosphate,
    BaseRibose,
    BPh,
    LeontisWesthof,
    Residue,
    Saenger,
    Stacking,
    StackingTopology,
    Structure2D,
)
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import (
    BASE_ACCEPTORS,
    BASE_ATOMS,
    BASE_DONORS,
    BASE_EDGES,
    PHOSPHATE_ACCEPTORS,
    RIBOSE_ACCEPTORS,
    Atom,
    Residue3D,
    Structure3D,
    torsion_angle,
)

HYDROGEN_BOND_MAX_DISTANCE = 4.0
HYDROGEN_BOND_MAX_PLANAR_DISTANCE = HYDROGEN_BOND_MAX_DISTANCE / 2.0
STACKING_MAX_DISTANCE = 6.0
STACKING_MAX_ANGLE_BETWEEN_NORMALS = 35.0
STACKING_MAX_ANGLE_BETWEEN_VECTOR_AND_NORMAL = 45.0


def compute_planar_distance(residue_i: Residue3D, atom_i: Atom, atom_j: Atom) -> float:
    normal_i = residue_i.base_normal_vector
    if normal_i is None:
        return math.inf

    vector_ij = atom_i.coordinates - atom_j.coordinates
    return numpy.linalg.norm(numpy.dot(normal_i, vector_ij)).item()


def angle_between_vectors(
    v1: numpy.typing.NDArray[numpy.floating], v2: numpy.typing.NDArray[numpy.floating]
) -> float:
    return math.acos(numpy.dot(v1, v2) / numpy.linalg.norm(v1) / numpy.linalg.norm(v2))


def detect_cis_trans(residue_i: Residue3D, residue_j: Residue3D) -> Optional[str]:
    c1p_i = residue_i.find_atom("C1'")
    c1p_j = residue_j.find_atom("C1'")

    if residue_i.one_letter_name in "AG":
        n9n1_i = residue_i.find_atom("N9")
    else:
        n9n1_i = residue_i.find_atom("N1")

    if residue_j.one_letter_name in "AG":
        n9n1_j = residue_j.find_atom("N9")
    else:
        n9n1_j = residue_j.find_atom("N1")

    if c1p_i is None or c1p_j is None or n9n1_i is None or n9n1_j is None:
        return None

    torsion = math.degrees(torsion_angle(c1p_i, n9n1_i, n9n1_j, c1p_j))
    return "c" if -90.0 < torsion < 90.0 else "t"


def detect_saenger(
    residue_i: Residue3D, residue_j: Residue3D, lw: LeontisWesthof
) -> Optional[Saenger]:
    key = (f"{residue_i.one_letter_name}{residue_j.one_letter_name}", lw.value)
    if key in Saenger.table():
        return Saenger[Saenger.table()[key]]
    return None


def detect_bph_br_classification(
    donor_residue: Residue3D, donor: Atom, acceptor: Atom
) -> Optional[int]:
    # source: Classification and energetics of the base-phosphate interactions in RNA. Craig L. Zirbel, Judit E. Sponer, Jiri Sponer, Jesse Stombaugh and Neocles B. Leontis
    if donor_residue.one_letter_name == "A":
        if donor.name == "C2":
            return 2
        if donor.name == "N6":
            n1 = donor_residue.find_atom("N1")
            c6 = donor_residue.find_atom("C6")
            if n1 is not None and c6 is not None:
                torsion = math.degrees(torsion_angle(n1, c6, donor, acceptor))
                return 6 if -90.0 < torsion < 90.0 else 7
        if donor.name == "C8":
            return 0

    if donor_residue.one_letter_name == "G":
        if donor.name == "N1":
            return 5
        if donor.name == "N2":
            n3 = donor_residue.find_atom("N3")
            c2 = donor_residue.find_atom("C2")
            if n3 is not None and c2 is not None:
                torsion = math.degrees(torsion_angle(n3, c2, donor, acceptor))
                return 1 if -90.0 < torsion < 90.0 else 3
        if donor.name == "C8":
            return 0

    if donor_residue.one_letter_name == "C":
        if donor.name == "N4":
            n3 = donor_residue.find_atom("N3")
            c4 = donor_residue.find_atom("C4")
            if n3 is not None and c4 is not None:
                torsion = math.degrees(torsion_angle(n3, c4, donor, acceptor))
                return 6 if -90.0 < torsion < 90.0 else 7
        if donor.name == "C5":
            return 9
        if donor.name == "C6":
            return 0

    if donor_residue.one_letter_name == "U":
        if donor.name == "N3":
            return 5
        if donor.name == "C5":
            return 9
        if donor.name == "C6":
            return 0

    if donor_residue.one_letter_name == "T":
        if donor.name == "N3":
            return 5
        if donor.name == "C6":
            return 0
        if donor.name == "C7":
            return 9

    return None


def merge_and_clean_bph_br(
    pairs: List[Tuple[Residue3D, Residue3D, int]]
) -> Dict[Tuple[Residue3D, Residue3D], List[int]]:
    bph_br_map = defaultdict(list)
    for residue_i, residue_j, classification in pairs:
        bph_br_map[(residue_i, residue_j)].append(classification)
    for bphs_brs in bph_br_map.values():
        # 3BPh and 5BPh simultanously means that it is actually 4BPh
        if 3 in bphs_brs and 5 in bphs_brs:
            bphs_brs.remove(3)
            bphs_brs.remove(5)
            bphs_brs.append(4)
        # 7BPh and 9BPh simultanously means that it is actually 8BPh
        if 7 in bphs_brs and 9 in bphs_brs:
            bphs_brs.remove(7)
            bphs_brs.remove(9)
            bphs_brs.append(8)
    return bph_br_map


def find_pairs(
    structure: Structure3D, model: int = 1
) -> Tuple[List[BasePair], List[BasePhosphate], List[BaseRibose]]:
    # put all donors and acceptors into a KDTree
    coordinates = []
    coordinates_atom_map: Dict[Tuple[float, float, float], Atom] = {}
    coordinates_type_map: Dict[Tuple[float, float, float], str] = {}
    coordinates_residue_map: Dict[Tuple[float, float, float], Residue3D] = {}
    for residue in structure.residues:
        if residue.model != model:
            continue
        acceptors = (
            BASE_ACCEPTORS.get(residue.one_letter_name, [])
            + RIBOSE_ACCEPTORS
            + PHOSPHATE_ACCEPTORS
        )
        donors = BASE_DONORS.get(residue.one_letter_name, [])
        for atom_name in acceptors + donors:
            atom = residue.find_atom(atom_name)
            if atom:
                xyz = (atom.x, atom.y, atom.z)
                coordinates.append(xyz)
                coordinates_atom_map[xyz] = atom
                coordinates_type_map[xyz] = (
                    "acceptor" if atom_name in acceptors else "donor"
                )
                coordinates_residue_map[xyz] = residue
    kdtree = KDTree(coordinates)

    # find all hydrogen bonds
    hydrogen_bonds = []
    base_phosphate_pairs = []
    base_ribose_pairs = []
    for i, j in kdtree.query_pairs(HYDROGEN_BOND_MAX_DISTANCE):
        type_i = coordinates_type_map[coordinates[i]]
        type_j = coordinates_type_map[coordinates[j]]

        # process only acceptor/donor pairs, not acceptor/acceptor or donor/donor
        if type_i == type_j:
            continue

        atom_i = coordinates_atom_map[coordinates[i]]
        atom_j = coordinates_atom_map[coordinates[j]]

        # skip spurious hydrogen bonds in the same residue
        if (
            atom_i.label is not None
            and atom_i.label is not None
            and atom_i.label == atom_j.label
        ):
            continue
        if (
            atom_i.auth is not None
            and atom_i.auth is not None
            and atom_i.auth == atom_j.auth
        ):
            continue

        residue_i = coordinates_residue_map[coordinates[i]]
        residue_j = coordinates_residue_map[coordinates[j]]

        # check for base-phosphate contacts
        if atom_i.name in PHOSPHATE_ACCEPTORS or atom_j.name in PHOSPHATE_ACCEPTORS:
            if type_i == "donor":
                donor_residue, acceptor_residue = residue_i, residue_j
                donor_atom, acceptor_atom = atom_i, atom_j
            else:
                donor_residue, acceptor_residue = residue_j, residue_i
                donor_atom, acceptor_atom = atom_j, atom_i
            planar_distance = compute_planar_distance(
                donor_residue, donor_atom, acceptor_atom
            )
            if planar_distance > HYDROGEN_BOND_MAX_PLANAR_DISTANCE:
                continue
            bph = detect_bph_br_classification(donor_residue, donor_atom, acceptor_atom)
            if bph is not None:
                base_phosphate_pairs.append((donor_residue, acceptor_residue, bph))
            continue

        # check for base-ribose contacts
        if atom_i.name in RIBOSE_ACCEPTORS or atom_j.name in RIBOSE_ACCEPTORS:
            if type_i == "donor":
                donor_residue, acceptor_residue = residue_i, residue_j
                donor_atom, acceptor_atom = atom_i, atom_j
            else:
                donor_residue, acceptor_residue = residue_j, residue_i
                donor_atom, acceptor_atom = atom_j, atom_i
            planar_distance = compute_planar_distance(
                donor_residue, donor_atom, acceptor_atom
            )
            if planar_distance > HYDROGEN_BOND_MAX_PLANAR_DISTANCE:
                continue
            br = detect_bph_br_classification(donor_residue, donor_atom, acceptor_atom)
            if br is not None:
                base_ribose_pairs.append((donor_residue, acceptor_residue, br))
            continue

        # check for base-base contacts
        planar_distance = min(
            [
                compute_planar_distance(residue_i, atom_i, atom_j),
                compute_planar_distance(residue_j, atom_i, atom_j),
            ]
        )
        if planar_distance < HYDROGEN_BOND_MAX_PLANAR_DISTANCE:
            hydrogen_bonds.append((atom_i, atom_j, residue_i, residue_j))

    # match hydrogen bonds with base edges
    labels = []
    for atom_i, atom_j, residue_i, residue_j in hydrogen_bonds:
        edges_i = BASE_EDGES.get(residue_i.one_letter_name, dict()).get(
            atom_i.name, None
        )
        edges_j = BASE_EDGES.get(residue_j.one_letter_name, dict()).get(
            atom_j.name, None
        )
        if edges_i is None or edges_j is None:
            continue

        # detect cis/trans
        cis_trans = detect_cis_trans(residue_i, residue_j)
        if cis_trans is None:
            continue

        if residue_i < residue_j:
            for edge_i in edges_i:
                for edge_j in edges_j:
                    labels.append((residue_i, residue_j, cis_trans, edge_i, edge_j))
        else:
            for edge_i in edges_i:
                for edge_j in edges_j:
                    labels.append((residue_j, residue_i, cis_trans, edge_j, edge_i))

    # create a list of base pairs
    base_base_pairs = []
    occupied = set()

    counter = Counter(labels)
    for interaction, hydrogen_bond_count in counter.most_common():
        if hydrogen_bond_count < 2:
            continue

        residue_i, residue_j, cis_trans, edge_i, edge_j = interaction

        if (residue_i, edge_i) in occupied:
            continue
        if (residue_j, edge_j) in occupied:
            continue

        occupied.add((residue_i, edge_i))
        occupied.add((residue_j, edge_j))

        lw = LeontisWesthof[f"{cis_trans}{edge_i}{edge_j}"]
        base_base_pairs.append((residue_i, residue_j, lw))

    base_pairs = []
    for residue_i, residue_j, lw in sorted(base_base_pairs):
        base_pairs.append(
            BasePair(
                Residue(residue_i.label, residue_i.auth),
                Residue(residue_j.label, residue_j.auth),
                lw,
                detect_saenger(residue_i, residue_j, lw),
            )
        )

    bph_map = merge_and_clean_bph_br(sorted(base_phosphate_pairs))
    base_phosphates = []
    for pair, bphs in bph_map.items():
        residue_i, residue_j = pair
        for bph in bphs:
            base_phosphates.append(
                BasePhosphate(
                    Residue(residue_i.label, residue_i.auth),
                    Residue(residue_j.label, residue_j.auth),
                    BPh[f"_{bph}"],
                )
            )

    br_map = merge_and_clean_bph_br(sorted(base_ribose_pairs))
    base_riboses = []
    for pair, brs in br_map.items():
        residue_i, residue_j = pair
        for br in brs:
            base_riboses.append(
                BaseRibose(
                    Residue(residue_i.label, residue_i.auth),
                    Residue(residue_j.label, residue_j.auth),
                    BR[f"_{br}"],
                )
            )

    return base_pairs, base_phosphates, base_riboses


def find_stackings(structure: Structure3D, model: int = 1) -> List[Stacking]:
    # put all nitrogen ring centers into a KDTree
    coordinates = []
    coordinates_residue_map: Dict[Tuple[float, float, float], Residue3D] = {}
    for residue in structure.residues:
        if residue.model != model:
            continue
        base_atoms = BASE_ATOMS.get(residue.one_letter_name, [])
        xs, ys, zs = [], [], []
        for atom_name in base_atoms:
            atom = residue.find_atom(atom_name)
            if atom is not None:
                xs.append(atom.x)
                ys.append(atom.y)
                zs.append(atom.z)
        if len(xs) > 0:
            geometric_center = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
            coordinates.append(geometric_center)
            coordinates_residue_map[geometric_center] = residue
    kdtree = KDTree(coordinates)

    # find all stacking interaction
    pairs = []
    for i, j in kdtree.query_pairs(STACKING_MAX_DISTANCE):
        residue_i = coordinates_residue_map[coordinates[i]]
        residue_j = coordinates_residue_map[coordinates[j]]

        # check angle between normals
        normal_i = residue_i.base_normal_vector
        normal_j = residue_j.base_normal_vector
        if normal_i is None or normal_j is None:
            continue

        angle = min(
            [
                angle_between_vectors(normal_i, normal_j),
                angle_between_vectors(-normal_i, normal_j),
            ]
        )
        if math.degrees(angle) > STACKING_MAX_ANGLE_BETWEEN_NORMALS:
            continue

        vector = numpy.array([coordinates[i][k] - coordinates[j][k] for k in (0, 1, 2)])
        angle = min(
            angle_between_vectors(vector, normal_i),
            angle_between_vectors(vector, normal_j),
        )
        if math.degrees(angle) > STACKING_MAX_ANGLE_BETWEEN_VECTOR_AND_NORMAL:
            continue

        same_direction = True if numpy.dot(normal_i, normal_j) > 0.0 else False

        if residue_i < residue_j:
            if same_direction:
                pairs.append((residue_i, residue_j, "upward"))
            else:
                pairs.append((residue_i, residue_j, "inward"))
        else:
            if same_direction:
                pairs.append((residue_j, residue_i, "downward"))
            else:
                pairs.append((residue_j, residue_i, "outward"))

    stackings = []
    for residue_i, residue_j, topology in sorted(pairs):
        nt1 = Residue(residue_i.label, residue_i.auth)
        nt2 = Residue(residue_j.label, residue_j.auth)
        stackings.append(Stacking(nt1, nt2, StackingTopology[topology]))

    return stackings


def extract_secondary_structure(
    tertiary_structure: Structure3D, model: int = 1
) -> Structure2D:
    base_pairs, base_phosphate, base_ribose = find_pairs(tertiary_structure, model)
    stackings = find_stackings(tertiary_structure, model)
    return Structure2D(base_pairs, stackings, base_ribose, base_phosphate, [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to PDB or mmCIF file")
    args = parser.parse_args()

    with open(args.input) as f:
        structure3d = read_3d_structure(f, 1)

    structure2d = extract_secondary_structure(structure3d)
    print(orjson.dumps(structure2d).decode("utf-8"))


if __name__ == "__main__":
    main()
