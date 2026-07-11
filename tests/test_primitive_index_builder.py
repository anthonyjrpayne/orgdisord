import numpy as np
import pytest
from ase import Atoms

from orgdisord.enumerate import OrderedfromDisordered


def _make_minimal_enumerator():
    """Construct only the state required by the builder."""
    enumerator = object.__new__(OrderedfromDisordered)
    primitive_cell = np.diag([10.0, 11.0, 12.0])

    # This molecule deliberately crosses the +a primitive-cell boundary.
    image_0 = Atoms(
        symbols=["C", "C"],
        positions=[
            [9.6, 1.0, 1.0],
            [10.8, 1.0, 1.0],
        ],
        cell=primitive_cell,
        pbc=False,
    )

    image_1 = Atoms(
        symbols=["N", "N"],
        positions=[
            [2.0, 2.0, 2.0],
            [3.2, 2.0, 2.0],
        ],
        cell=primitive_cell,
        pbc=False,
    )

    enumerator.cell = primitive_cell
    enumerator.supercell = [2, 1, 1]
    enumerator.images = [image_0, image_1]

    return enumerator


def test_builder_preserves_whole_molecule_coordinates():
    enumerator = _make_minimal_enumerator()

    result = enumerator.build_supercell_from_primitive_indices([0, 1])

    assert len(result) == 4
    assert np.allclose(
        result.cell.array,
        np.diag([20.0, 11.0, 12.0]),
    )

    # The first image must not be wrapped atom-by-atom.
    assert np.allclose(
        result.positions[:2],
        enumerator.images[0].positions,
    )

    expected_second_image = (
        enumerator.images[1].positions
        + np.array([10.0, 0.0, 0.0])
    )
    assert np.allclose(
        result.positions[2:],
        expected_second_image,
    )

    # Intramolecular vectors are unchanged.
    assert np.allclose(
        result.positions[1] - result.positions[0],
        np.array([1.2, 0.0, 0.0]),
    )


def test_builder_rejects_wrong_number_of_indices():
    enumerator = _make_minimal_enumerator()

    with pytest.raises(ValueError, match="expected 2"):
        enumerator.build_supercell_from_primitive_indices([0])


def test_builder_rejects_invalid_index():
    enumerator = _make_minimal_enumerator()

    with pytest.raises(IndexError, match="valid range"):
        enumerator.build_supercell_from_primitive_indices([0, 99])


def test_group_choices_are_converted_cell_by_cell():
    enumerator = _make_minimal_enumerator()

    class MinimalDisorderedStructure:
        Z = 2

    enumerator.disordered_structure = MinimalDisorderedStructure()
    enumerator.ndisordergroups = 2

    # Two cells, each containing two binary group choices:
    # [0, 1] -> primitive index 1; [1, 0] -> primitive index 2.
    assert enumerator._primitive_indices_from_group_choices(
        [0, 1, 1, 0]
    ) == [1, 2]


def test_group_choice_conversion_rejects_wrong_length():
    enumerator = _make_minimal_enumerator()

    class MinimalDisorderedStructure:
        Z = 2

    enumerator.disordered_structure = MinimalDisorderedStructure()
    enumerator.ndisordergroups = 2

    with pytest.raises(ValueError, match="divided evenly"):
        enumerator._primitive_indices_from_group_choices([0, 1, 1])
