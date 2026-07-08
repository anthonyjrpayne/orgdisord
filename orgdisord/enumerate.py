import numpy as np
import os
import re
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.spacegroup.spacegroup import SpacegroupValueError
from ase import Atoms, Atom
from ase.data import covalent_radii
from ase.geometry import get_duplicate_atoms
from soprano.properties.linkage import Molecules
from collections import Counter
import itertools
from tqdm import tqdm
from scipy.spatial.distance import pdist
import logging
from functools import lru_cache
from orgdisord.utils import reload_as_molecular_crystal, random_product


logger = logging.getLogger("orgdisord.enumerate")


def _split_label(label):
    """Return (component, site, atom_name) from labels like A2m1_S001_C13."""
    text = str(label)
    parts = text.split("_")

    component = None
    site = None
    atom_name = text

    if parts:
        match = re.match(r"(A\d+)", parts[0])
        component = match.group(1) if match else parts[0]

    for part in parts:
        if re.fullmatch(r"S\d+", part):
            site = part
            break

    if len(parts) >= 2:
        atom_name = parts[-1]

    return component, site, atom_name


def _atoms_labels(atoms):
    """Return atom labels if present, otherwise stable symbol-index fallback labels."""
    if atoms.has("labels"):
        return np.asarray(atoms.get_array("labels")).astype(str)

    return np.asarray(
        [f"{symbol}{i}" for i, symbol in enumerate(atoms.get_chemical_symbols())],
        dtype=str,
    )


def _same_molecular_site(labels, i, j):
    """
    Only allow bonding within the same labelled site when site labels exist.

    This prevents neighbour-list connectivity joining different disordered
    molecular sites across a short intermolecular contact. If site labels are
    absent, fall back to normal chemical connectivity.
    """
    si = _split_label(labels[i])[1]
    sj = _split_label(labels[j])[1]

    if si is None or sj is None:
        return True

    return si == sj


def connected_components_indices(atoms, cutoff_scale=1.5, exclude_hh=True):
    """Return bonded molecular components as lists of atom indices."""
    labels = _atoms_labels(atoms)
    numbers = atoms.get_atomic_numbers().astype(int)

    radii = np.asarray(
        [float(covalent_radii[z]) if z < len(covalent_radii) else 0.7 for z in numbers]
    )
    nl = NeighborList(
        (radii * cutoff_scale).tolist(),
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    nl.update(atoms)

    visited = np.zeros(len(atoms), dtype=bool)
    components = []

    for start in range(len(atoms)):
        if visited[start]:
            continue

        queue = [start]
        visited[start] = True
        component = []

        while queue:
            i = queue.pop(0)
            component.append(i)

            neighbours, offsets = nl.get_neighbors(i)
            for j in neighbours:
                if exclude_hh and numbers[i] == 1 and numbers[j] == 1:
                    continue
                if not _same_molecular_site(labels, i, int(j)):
                    continue
                if not visited[j]:
                    visited[j] = True
                    queue.append(int(j))

        components.append(component)

    return components


def make_whole_molecule(atoms, cutoff_scale=1.5, preserve_pbc=False, exclude_hh=True):
    """
    Unwrap bonded molecules so each connected component is Cartesian-contiguous.

    This is the only coordinate repair done here. It deliberately does not sort
    atoms, standardise molecular orientation, or choose a different molecular
    image by shape matching.
    """
    if len(atoms) == 0:
        return atoms.copy()

    A = atoms.copy()
    labels = _atoms_labels(A)
    numbers = A.get_atomic_numbers().astype(int)
    cell = np.asarray(A.get_cell())
    positions = A.get_positions().copy()

    radii = np.asarray(
        [float(covalent_radii[z]) if z < len(covalent_radii) else 0.7 for z in numbers]
    )
    nl = NeighborList(
        (radii * cutoff_scale).tolist(),
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    nl.update(A)

    visited = np.zeros(len(A), dtype=bool)

    for start in range(len(A)):
        if visited[start]:
            continue

        queue = [start]
        visited[start] = True

        while queue:
            i = queue.pop(0)
            pi = positions[i]

            neighbours, offsets = nl.get_neighbors(i)
            for nb_pos, j in enumerate(neighbours):
                j = int(j)

                if exclude_hh and numbers[i] == 1 and numbers[j] == 1:
                    continue
                if not _same_molecular_site(labels, i, j):
                    continue
                if visited[j]:
                    continue

                ri = float(covalent_radii[numbers[i]])
                rj = float(covalent_radii[numbers[j]])
                bond_cutoff = cutoff_scale * (ri + rj)

                raw_pos = positions[j]
                if np.linalg.norm(raw_pos - pi) <= bond_cutoff + 1e-6:
                    best_pos = raw_pos
                else:
                    best_pos = raw_pos + np.asarray(offsets[nb_pos], dtype=float) @ cell

                positions[j] = best_pos
                visited[j] = True
                queue.append(j)

    A.set_positions(positions)
    if not preserve_pbc:
        A.set_pbc([False, False, False])

    return A


def save_as_xyz_raw(atoms_object, filename, debug_name=""):
    """Save coordinates exactly as stored in the ASE Atoms object."""
    with open(filename, "w") as handle:
        handle.write(f"{len(atoms_object)}\n")
        handle.write(f"RAW State ({debug_name})\n")
        for symbol, pos in zip(atoms_object.get_chemical_symbols(), atoms_object.positions):
            handle.write(f"{symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")


def save_primitive_cells(primitive_configs, primitive_dir="primitive_cells"):
    """Write primitive configurations for debugging/review."""
    os.makedirs(primitive_dir, exist_ok=True)

    for i_config, atoms in enumerate(primitive_configs):
        save_as_xyz_raw(
            atoms,
            os.path.join(primitive_dir, f"primitive_config_{i_config}.xyz"),
            debug_name=f"primitive_config_{i_config}",
        )

    logger.info("Wrote %d primitive configurations to %s", len(primitive_configs), os.path.abspath(primitive_dir))


def binary_to_idx(config, maxgroups):
    return sum([j * (maxgroups**i) for i, j in list(enumerate(reversed(config)))])

def idx_to_config(index, maxgroups, length):
    """
    Inverse of binary_to_idx().

    Example for maxgroups=2, length=2:
        0 -> (0, 0)
        1 -> (0, 1)
        2 -> (1, 0)
        3 -> (1, 1)
    """
    index = int(index)

    digits = [0] * length

    for i in range(length - 1, -1, -1):
        digits[i] = index % maxgroups
        index //= maxgroups

    if index != 0:
        raise ValueError(
            f"Primitive config index is too large for maxgroups={maxgroups}, "
            f"length={length}"
        )

    return tuple(digits)


def select_configs(config, supercell, maxgroups):
    """
    from the supercell config, return indices of each
    component primitive cell
    e.g.
    a 2x2x1 supercell of a Z=4 system
    would have configs of length 16.
    This function chunks them into 4
    sets of 4 and returns the integer representation
    (0-15) of each 'binary' number.
    So for example:
    [1,1,1,1, 0,0,0,0, 1,1,0,0, 0,0,0,1]
    would return
    [15, 0, 12, 1]

    """
    na, nb, nc = supercell
    nsuper = na * nb * nc
    chunk = len(config) // nsuper
    # should equal Z
    return [binary_to_idx(c, maxgroups) for c in chunks(config, chunk)]


def chunks(lst, n, offset=0):
    """Yield successive n-sized chunks from lst."""
    if offset != 0:
        # rotate copy of list by offset
        temp = lst[offset:] + lst[:offset]
    else:
        temp = lst
    for i in np.arange(0, len(lst), n):
        yield temp[i : i + n]


def merge_component_counts(*count_dicts):
    counts = Counter()
    for count_dict in count_dicts:
        counts.update(count_dict)
    return dict(counts)


def ratio_from_component_counts(component_counts):
    total = sum(component_counts.values())
    if total == 0:
        return {}
    return {
        key: value / total
        for key, value in sorted(component_counts.items())
    }


class OrderedfromDisordered:
    def __init__(self, disordered_structure, symprec=1e-4, quiet=False):
        """
        Args:
            disordered_structure (DisorderedStructure)
            symprec (float): tolerance for symmetry finding
            quiet (bool): if True, suppresses warnings

        """
        self.quiet = quiet
        self.disordered_structure = disordered_structure

        self.symprec = symprec
        self.logger = logging.getLogger("orgdisord.enumerate")
        self.logger.debug("\n\n")
        self.logger.debug("------------------------------")
        self.logger.debug("--- ENUMERATING STRUCTURES ---")
        self.logger.debug("------------------------------")

    def _get_config_symbols_coords(self, config, asslabel):
        """
        takes config as an int (often binary) list
        of length Z (number of disordered molecular units in cell)
        and the index of the chosen assembly

        Args:
            config (list): list of integers representing the config
            asslabel (str): Assembly label e.g. 'A'

        Returns:
            [
                symbols (list): list of symbols for the config
                sites (list): list of sites for the config
                tags (list): list of tags for the config
                labels (list): list of labels for the config
            ]

        """
        assembly = self.disordered_structure.get_assembly(asslabel)
        disorder_groups = assembly.disorder_groups
        # disorder_group_labels = sorted(group.label for group in disorder_groups)

        # a few aliases
        # how many disorder groups in this assembly?
        ndisordergroups = len(disorder_groups)

        sites = []
        symbols = []
        tags = []
        labels = []
        # loop over groups
        for igroup in range(ndisordergroups):
            group = disorder_groups[igroup]
            assert len(group.atoms) > 0
            group_symmops = group.symmetry_operations

            if len(group_symmops) == len(config):
                # choose the subset of symmops for this group given config
                group_config_symmops = [
                    group_symmops[i] for i, c in enumerate(config) if c == igroup
                ]
            else:
                # Zprime < 1 so config instead needs to index into subset of group symops
                # e.g. config [0,0,1,0] for would select those subsets of symops for this group
                # probably need to select the inverse for the next group...

                group_config_symmops = [
                    group_symmops[c][i] for i, c in enumerate(config)
                ]

            # tag = igroup + 1 if nassemblies == 1 else 1e2*(iassembly+1) + igroup + 1
            tag = group.tag
            [symbols_g, sites_g, tags_g, labels_g] = self._get_group_config_sites(
                tag, group, group_config_symmops
            )
            symbols += symbols_g
            sites += sites_g
            tags += tags_g
            labels += labels_g

        return [symbols, sites, tags, labels]

    def _get_group_config_sites(self, tag, group, group_config_symmops):
        """
        Apply symmetry operations to a disorder group.

        Important:
            We preserve the unwrapped fractional coordinates from the CIF.
            This allows molecules deliberately placed partly outside [0, 1)
            to remain on the intended side of the unit cell.

            Wrapped coordinates are used only for duplicate/equivalence checks.
        """
        sites = []
        wrapped_sites = []
        symbols = []
        tags = []
        labels = []

        unique_positions = group.atoms.get_scaled_positions(wrap=False)
        unique_symbols = group.atoms.get_chemical_symbols()
        unique_labels = group.atoms.get_array("labels")

        for op in group_config_symmops:
            rot, trans = op

            for kind, pos in enumerate(unique_positions):
                # Preserve the raw transformed fractional coordinate.
                # Do NOT np.mod this.
                site = np.dot(rot, pos) + trans

                # Use wrapped version only to detect equivalent duplicate atoms.
                site_wrapped = np.mod(site, 1.0)

                if not wrapped_sites:
                    sites.append(site)
                    wrapped_sites.append(site_wrapped)
                    symbols.append(unique_symbols[kind])
                    tags.append(tag)
                    labels.append(unique_labels[kind])
                    continue

                t = site_wrapped - np.asarray(wrapped_sites)
                t -= np.rint(t)

                mask = np.all(np.abs(t) < self.symprec, axis=1)

                if np.any(mask):
                    inds = np.argwhere(mask).flatten()

                    if len(inds) > 1:
                        raise SpacegroupValueError("Found multiple equivalent sites!")

                    # Equivalent duplicate; skip adding it.
                    continue

                sites.append(site)
                wrapped_sites.append(site_wrapped)
                symbols.append(unique_symbols[kind])
                tags.append(tag)
                labels.append(unique_labels[kind])

        return [symbols, sites, tags, labels]

    def get_config(self, config, asslabel):
        """
        Return a fully ordered atoms object for the specified config.
        Output atoms are provided in Cartesian coordinates (absolute relative to the
        primitive cell origin) and are unwrapped by connectivity (so molecules are whole).
        """
        # set up base atoms object to which sites will be
        # added (just an empty box).
        atoms = Atoms(cell=self.disordered_structure.cell, pbc=True)

        # add in disordered sites, ordered according to config, (tag = 1e2*(iassembly+1) + igroup + 1)
        symbols, sites, tags, labels = self._get_config_symbols_coords(
            config, asslabel=asslabel
        )
        for site, symbol, tag in zip(sites, symbols, tags):
            atom = Atom(symbol=symbol, position=atoms.cell.T.dot(site), tag=tag)
            atoms.append(atom)
        atoms.set_array("labels", np.array(labels))

        # Unwrap the molecule based on bonding graph so bonded atoms are contiguous.
        # The returned atoms will have Cartesian positions possibly outside the unit cell.
        atoms = make_whole_molecule(atoms, preserve_pbc=False)

        return atoms

    def get_all_configs(
        self,
        exclude_ordered=False,
        keep_primitive=False,
        primitive_dir="primitive_cells",
        ):
        """Generate all possible configs for the primitive cell and return them as
        absolute-cartesian, unwrapped ASE Atoms objects (NOT wrapped into [0,1)).
        """

        cell = self.disordered_structure.cell
        Z = self.disordered_structure.Z
        # Zprime = self.cif.Zprime
        nops_all = self.disordered_structure.spacegroup.nsymop
        nassemblies = self.disordered_structure.get_number_of_assemblies()
        correlated_assemblies = self.disordered_structure.correlated_assemblies

        max_spacegroup_kinds = 0
        # set up base atoms object to which sites will be
        # added.
        if exclude_ordered:
            # just an empty box
            atoms = Atoms(cell=cell, pbc=True)
        else:
            atoms = self.disordered_structure.ordered_atoms.copy()
            if len(atoms) > 0:
                sg = self.disordered_structure.spacegroup
                max_spacegroup_kinds = sg.tag_sites(atoms.get_scaled_positions()).max()
                # give ordered sites a tag of 0
                atoms.set_tags(0)

        # what's the final number of configs we will generate?
        # TODO generalise!
        ngroups_per_assembly = (
            self.disordered_structure.get_number_of_disorder_groups_per_assembly()
        )
        nconfigs_per_assembly = [N**Z for N in ngroups_per_assembly]
        nconfigs = nconfigs_per_assembly[0]  # defaults to first ngroups

        if correlated_assemblies:
            if nassemblies > 1:
                self.logger.debug(
                    "Treating the assemblies as correlated. \n"
                    "This means that the same index from each assembly is chosen for each configuration."
                )
                if len(set(nconfigs_per_assembly)) != 1:
                    raise ValueError(
                        "Error: the number of configurations per assembly is not the same for all assemblies.\n"
                        "Please set correlated_assemblies to False."
                    )
        else:
            if nassemblies > 1:
                self.logger.debug(
                    "Treating the assemblies as uncorrelated "
                    "(i.e. the assemblies are independent) and we will therefore generate lots of structures!"
                )
                nconfigs = np.product(nconfigs_per_assembly)
        self.logger.debug(f"Generating {nconfigs} configs in primitive cell.")
        if nconfigs > 1e4:
            self.logger.warn(
                f"Warning: {nconfigs} is a large number of configs. "
                "This may take a while!"
            )

        all_configs = []
        all_config_metadata = []
        for assembly in self.disordered_structure.disorder_assemblies:
            asslabel = assembly.label
            # the disorder groups for this assembly
            disorder_groups = assembly.disorder_groups
            # normally ngroups = ndisorder groups
            ngroups = len(disorder_groups)
            group_nops = [len(group.symmetry_operations) for group in disorder_groups]
            self.logger.debug(f"Group n ops {group_nops}")

            if len(set(group_nops)) != 1:
                raise ValueError(
                    "Error: the number of symmetry operations per group "
                    f" is not the same for all groups in assembly {asslabel}:\n"
                    f"{group_nops}"
                )

            # but when Zprime is less than 1, we need to be more careful!
            # if Zprime < 1: # means more symmetry operations than we have Z
            #     # in this case the actual number of groups (1/Zprime) will be larger than the number of disorder groups
            #     assert 1/Zprime >= ngroups

            #     ngroups = int(1 / Zprime)
            #     self.logger.warn(f'Special case for nops != Z. (i.e. Zprime < 1)'
            #         f'nops = {nops_all} '
            #         f'Z = {Z}, '
            #         f'Zprime = {Zprime}\n'
            #         'This is less well-tested so proceed with caution.'
            #         'If you encounter any issues, please contact the developers.')

            self.logger.debug(f"Assembly {asslabel} has {ngroups} groups")

            if ngroups == 1:
                # Special disorder site case with only one group
                # -> each symmetry operation generated a new config.
                config_max = group_nops[0]
            else:
                config_max = ngroups

            # generate all possible configs for this assembly
            config_indices = list(
                itertools.product(list(range(config_max)), repeat=Z)
            )

            assembly_configs = [
                self.get_config(config_idx, asslabel)
                for config_idx in config_indices
            ]

            assembly_config_metadata = [
                {
                    "components": self._component_counts_for_config(
                        config_idx,
                        asslabel,
                    )
                }
                for config_idx in config_indices
            ]
            # update max_spacegroup_kinds tags
            for config in assembly_configs:
                labels = config.get_array("labels")
                u, spacegroup_kinds = np.unique(labels, return_inverse=True)
                config.set_array(
                    "spacegroup_kinds", spacegroup_kinds + max_spacegroup_kinds + 1
                )
            self.logger.debug(
                f"Assembly {asslabel} has {len(assembly_configs)} configs"
            )
            if all_configs:
                if correlated_assemblies:
                    # then just add the config atoms to the existing ones!
                    assert len(assembly_configs) == len(all_configs)
                    for iconfig in range(len(assembly_configs)):
                        temp = all_configs[iconfig].copy()
                        temp.extend(assembly_configs[iconfig])
                        all_configs[iconfig] = temp
                        
                        all_config_metadata[iconfig]["components"] = (
                            merge_component_counts(
                                all_config_metadata[iconfig]["components"],
                                assembly_config_metadata[iconfig]["components"],
                            )
                        )
                else:
                    ref_configs = [config.copy() for config in all_configs]
                    ref_metadata = [dict(meta) for meta in all_config_metadata]

                    all_configs = []
                    all_config_metadata = []

                    for iconfig in range(len(ref_configs)):
                        for jconfig in range(len(assembly_configs)):
                            temp = ref_configs[iconfig].copy()
                            temp.extend(assembly_configs[jconfig])
                            all_configs.append(temp)

                            components = merge_component_counts(
                                ref_metadata[iconfig]["components"],
                                assembly_config_metadata[jconfig]["components"],
                            )
                            all_config_metadata.append(
                                {"components": components}
                            )
            else:
                # must the be first assembly -- all_configs is empty so let's start from scratch
                 for config, metadata in zip(
                    assembly_configs,
                    assembly_config_metadata,
                ):
                    temp = atoms.copy()
                    temp.extend(config)
                    temp = make_whole_molecule(temp, preserve_pbc=False)
                    all_configs.append(temp)
                    all_config_metadata.append(metadata)

        self.primitive_config_metadata = all_config_metadata

        self.logger.debug(f"Generated {len(all_configs)} primitive configs")
        return all_configs

    def get_supercell_configs(
        self,
        supercell=(1, 1, 1),
        maxiters=512,
        exclude_ordered=False,
        random_configs=False,
        return_configs=False,
        fix_ratio=False,
        ratio=None,
        ratio_tolerance=0.01,
        keep_primitive=False,
        primitive_dir="primitive_cells",
        reload_molecular_crystal=False,
    ):
        """
        loop over supercell cells,
        add in one of the ndisordergroups^Z configurations per cell
        if return_configs is True, return the configs as well as the supercells
        if molecular_crystal is True, then the structure is reloaded as a molecular crystal
             -- trying to keep the pieces in tact

        Args:
            supercell (list/tuple): supercell dimensions e.g. [2,2,2]
            maxiters (int): maximum number of structures to generate. If random_configs is True,
                            then this is the number of random configs to generate.
                            If random_configs is False and the total number of configs is less than maxiters,
                            then all configs will be generated.
                            Default is 5000.
            exclude_ordered (bool): if True, exclude the ordered part of supercell in the returned structures.
            random_configs (bool): if True, choose the configs randomly from all allowed configs.
            return_configs (bool): if True, return the configs (e.g. (0,1,0,1)) as well as the supercells
            fix_ratio (bool): if True, then the ratio(s) of the disorder components is/are fixed. Will use the group
                              occupancies to determine the ratio(s) unless ratio is specified.
            ratio (float): Ratio(s) of the disorder components. If None, then the ratio(s) will be determined from the
                            group occupancies. If ratio is specified, then the group occupancies will be ignored.
                            If ratio is a float, then the ratio of the first group will be fixed to this value.
                            If ratio is a list, then the ratio of each group will be fixed to the corresponding value.
                            If ratio is a list of lists, then the ratio of each group will be fixed to the corresponding
                            value in the inner list. The outer list corresponds to the assembly.
                            Default is None.
            ratio_tolerance (float): Tolerance for the ratio(s) of the disorder components. Default is 0.01. See np.isclose atol parameter.

        Returns:
            supercells (list): list of supercells
            configs (list): list of configs (optional)
        """
        # pre-compute all primitive cells
        self.logger.debug("Pre-computing all primitive configurations...")
        self.ratio_tolerance = ratio_tolerance
        if ratio and not fix_ratio:
            self.logger.warn(
                "Warning! ratio is specified but fix_ratio is False. Ignoring ratio."
            )

        ########################################
        # Get all primitive configs
        self.images = self.get_all_configs(
            exclude_ordered=exclude_ordered,
            keep_primitive=keep_primitive,
            primitive_dir=primitive_dir,
        )
        
        self.component_labels = sorted(
            {
                component
                for metadata in self.primitive_config_metadata
                for component in metadata.get("components", {})
            },
            key=lambda x: int(x[1:]) if re.fullmatch(r"A\d+", str(x)) else str(x),
        )

        self.logger.info(
            "Component labels for ratio filtering: %s",
            self.component_labels,
        )
        
        self.raw_group_labels = self._raw_group_labels()
        nimages_prim = len(self.images)
        if self.disordered_structure.molecular_crystal:
            self.logger.debug(f"Reloading {nimages_prim} images as molecular crystals")
            cheap_method = False
            if nimages_prim > 256:
                # this seems to be get pretty slow for large numbers of images
                self.logger.warn(
                    "Warning: reloading molecular crystals is slow for this many images.\n"
                    "Consider skipping this step with the --not_molecular_crystal flag!\n"
                    "We will use a cheaper method to reload the images as molecular crystals instead\n"
                    " -- assuming the connectivity is the same for each image"
                )
                cheap_method = True

            if reload_molecular_crystal:
                try:
                    self.images = reload_as_molecular_crystal(
                        self.images,
                        cheap=cheap_method,
                    )
                except ValueError as exc:
                    logger.warning(
                        "Skipping reload_as_molecular_crystal because generated images "
                        "do not match a common molecular selection: %r",
                        exc,
                    )
        ########################################

        if keep_primitive:
            save_primitive_cells(
                self.images,
                primitive_dir=primitive_dir,
            )

        # some aliases
        Z = self.disordered_structure.Z
        self.cell = self.disordered_structure.cell
        na, nb, nc = supercell
        self.supercell = supercell
        # we previously just took ndisordergroups = self.cif.ndisordergroups, but
        # this gets tricky when we have multiple assemblies etc. Safer to just take this:
        self.ndisordergroups = int(np.round(np.exp(np.log(nimages_prim) / Z)))
        # total number of configs given this supercell:
        ncombinations = self.ndisordergroups ** (Z * na * nb * nc)
        self.logger.debug(
            f"""
        Found {nimages_prim} primitive configurations,
              {ncombinations} configs for this supercell (Z={Z}, na={na}, nb={nb}, nc={nc}),
              {self.ndisordergroups} ndisordergroups"""
        )

        options = list(range(self.ndisordergroups))

        n_primitive_cells = na * nb * nc
        n_raw_slots = Z * n_primitive_cells

        if fix_ratio:
            self.ratio = self._process_ratio(ratio)
            self.logger.info(f"Fixing ratio to {self.ratio}")

            target_ratio = np.asarray(self.ratio[0], dtype=float)

            use_direct_component_sampling = (
                random_configs
                and hasattr(self, "component_labels")
                and len(target_ratio) == len(self.component_labels)
            )
        else:
            use_direct_component_sampling = False

        if use_direct_component_sampling:
            all_combinations = self._random_component_ratio_configs(
                target_ratio=target_ratio,
                n_primitive_cells=n_primitive_cells,
                n_configs=maxiters,
            )

        elif random_configs:
            all_combinations = random_product(options, repeat=n_raw_slots)

            if fix_ratio:
                all_combinations = filter(self._ratio_filter, all_combinations)

        else:
            all_combinations = itertools.product(options, repeat=n_raw_slots)

            if fix_ratio:
                all_combinations = filter(self._ratio_filter, all_combinations)

        # how many configs to actually generate:
        if random_configs:
            # just take maxiters
            n_configs = maxiters
        else:
            # take whichever is smallest betwee
            # maxiters and ncombinations
            n_configs = min([maxiters, ncombinations])
            if n_configs < ncombinations:
                self.logger.warn(
                    f"""
                Warning: maxiters ({maxiters}) is smaller than the total number of configs ({ncombinations})
                We will only generate {n_configs} configs.
                You might want to consider increasing maxiters or using the --random flag"""
                )

        all_supercells = []
        all_configs = []
        all_supercell_metadata = []
        if random_configs:
            self.logger.info(
                f"Generating {n_configs} random configurations in a {supercell} supercell:"
            )
        else:
            self.logger.info(
                f"Generating {n_configs} out of the {ncombinations} possible configurations in the {supercell} supercell:"
            )

        for i in tqdm(range(n_configs), disable=self.quiet):
            config = next(all_combinations, False)
            if not config:
                self.logger.debug("No more configs left!")
                break
            # make ncells copies of the relevant config
            self.logger.debug(f"         {config}")
            all_supercells.append(self.get_supercell_config(config))
            all_configs.append(config)

            primitive_indices = select_configs(
                config,
                supercell=self.supercell,
                maxgroups=self.ndisordergroups,
            )

            component_counts = Counter()
            for prim_idx in primitive_indices:
                component_counts.update(
                    self.primitive_config_metadata[prim_idx]["components"]
                )

            component_counts = dict(sorted(component_counts.items()))
            all_supercell_metadata.append(
                {
                    "components": component_counts,
                    "ratio": ratio_from_component_counts(component_counts),
                    "multiplicity": 1,
                    "config": list(config),
                    "primitive_config_indices": list(primitive_indices),
                }
            )
            # TODO
            # check for overlapping atoms
            # cutoff = 0.25
            # dists = pdist(supercell_atoms.get_positions(), 'sqeuclidean')
            # if (dists < cutoff**2).any():
            #     continue
            # else:

        if fix_ratio:
            self.logger.info(
                f"Only {len(all_configs)}/{n_configs} configs had the specified ratio of {self.ratio}"
            )
            
        self.supercell_config_metadata = all_supercell_metadata
        
        
        if return_configs:
            return all_supercells, all_configs
        return all_supercells
    

    def _process_ratio(self, ratio):
        """
        Process the ratio argument.

        Supported forms:
            None
                use CIF group occupancies

            0.75
                legacy binary mode -> [[0.75, 0.25]]

            [0.5, 0.25, 0.25]
                explicit ratio vector

            [[...], [...]]
                explicit per-assembly ratio vectors
        """
        if ratio is None:
            ratio = [
                [g.occupancy for g in ass.disorder_groups]
                for ass in self.disordered_structure.disorder_assemblies
            ]

        elif isinstance(ratio, (float, int, np.floating, np.integer)):
            ratio = [[float(ratio), 1.0 - float(ratio)]]

        elif isinstance(ratio, np.ndarray):
            ratio = ratio.tolist()

        elif isinstance(ratio, (list, tuple)):
            ratio = list(ratio)

            if len(ratio) == 1 and isinstance(
                ratio[0],
                (float, int, np.floating, np.integer),
            ):
                value = float(ratio[0])
                ratio = [[value, 1.0 - value]]

            elif all(
                isinstance(value, (float, int, np.floating, np.integer))
                for value in ratio
            ):
                values = np.asarray(ratio, dtype=float)

                if values.sum() <= 0:
                    raise ValueError(f"Ratio values sum to zero: {ratio}")

                if not np.isclose(values.sum(), 1.0):
                    values = values / values.sum()

                ratio = [values.tolist()]

            elif isinstance(ratio[0], (list, tuple, np.ndarray)):
                ratio = [
                    np.asarray(inner, dtype=float).tolist()
                    for inner in ratio
                ]

            else:
                raise ValueError("ratio must be a float, list of floats, or list of lists")

        else:
            raise ValueError("ratio must be a float, list of floats, or list of lists")

        return np.asarray(ratio, dtype=float)

    def _primitive_component_vectors(self):
        """
        Return component count vectors for each primitive configuration.

        Example with component_labels ['A1', 'A2']:
            primitive config 0 -> (2, 0)
            primitive config 1 -> (1, 1)
            primitive config 8 -> (0, 2)
        """
        vectors = []

        for metadata in self.primitive_config_metadata:
            counts = metadata.get("components", {})

            vectors.append(
                tuple(
                    int(counts.get(component, 0))
                    for component in self.component_labels
                )
            )

        return vectors


    def _target_counts_from_ratio(self, target_ratio, n_primitive_cells):
        """
        Convert a target component ratio into integer component counts.

        n_primitive_cells is na * nb * nc.
        """
        vectors = self._primitive_component_vectors()

        sites_per_primitive = sum(vectors[0])
        total_sites = int(n_primitive_cells * sites_per_primitive)

        target_ratio = np.asarray(target_ratio, dtype=float)

        if target_ratio.sum() <= 0:
            raise ValueError(f"Target ratio sums to zero: {target_ratio}")

        target_ratio = target_ratio / target_ratio.sum()

        raw = target_ratio * total_sites

        counts = np.floor(raw).astype(int)
        remaining = total_sites - int(counts.sum())

        remainders = raw - counts
        order = np.argsort(remainders)[::-1]

        for idx in order[:remaining]:
            counts[idx] += 1

        self.logger.info(
            "Target component counts for %d primitive cells / %d sites: %s",
            n_primitive_cells,
            total_sites,
            dict(zip(self.component_labels, counts)),
        )

        return tuple(int(x) for x in counts)


    def _random_config_with_component_counts(self, target_counts, n_primitive_cells, rng):
        """
        Generate one random raw group-state config with exact component counts.

        The internal dynamic-programming step chooses primitive configuration
        indices, but the returned config is converted back to the old raw
        group-state representation expected by get_supercell_config().
        """
        vectors = self._primitive_component_vectors()
        n_components = len(self.component_labels)

        target_counts = tuple(int(x) for x in target_counts)

        @lru_cache(None)
        def feasible(cells_left, remaining):
            remaining = tuple(remaining)

            if cells_left == 0:
                return all(x == 0 for x in remaining)

            for vector in vectors:
                new_remaining = tuple(
                    remaining[i] - vector[i]
                    for i in range(n_components)
                )

                if any(x < 0 for x in new_remaining):
                    continue

                if feasible(cells_left - 1, new_remaining):
                    return True

            return False

        if not feasible(n_primitive_cells, target_counts):
            raise ValueError(
                "Could not build a supercell configuration with the requested "
                f"component counts {dict(zip(self.component_labels, target_counts))}. "
                "Try a larger supercell or a composition that is representable."
            )

        raw_config = []
        remaining = target_counts

        for cells_left in range(n_primitive_cells, 0, -1):
            choices = []

            for primitive_index, vector in enumerate(vectors):
                new_remaining = tuple(
                    remaining[i] - vector[i]
                    for i in range(n_components)
                )

                if any(x < 0 for x in new_remaining):
                    continue

                if feasible(cells_left - 1, new_remaining):
                    choices.append((primitive_index, new_remaining))

            choice_index = int(rng.integers(len(choices)))
            primitive_index, remaining = choices[choice_index]

            primitive_digits = idx_to_config(
                primitive_index,
                maxgroups=self.ndisordergroups,
                length=self.disordered_structure.Z,
            )

            raw_config.extend(primitive_digits)

        return tuple(raw_config)


    def _random_component_ratio_configs(self, target_ratio, n_primitive_cells, n_configs):
        """
        Yield random raw group-state configs matching a component-level ratio exactly.
        """
        rng = np.random.default_rng()

        target_counts = self._target_counts_from_ratio(
            target_ratio=target_ratio,
            n_primitive_cells=n_primitive_cells,
        )

        self.logger.info(
            "Directly generating random configs with component counts %s",
            dict(zip(self.component_labels, target_counts)),
        )

        for _ in range(n_configs):
            yield self._random_config_with_component_counts(
                target_counts=target_counts,
                n_primitive_cells=n_primitive_cells,
                rng=rng,
            )
    
    
    def _ratio_filter(self, config):
        """
        Return True if the config matches the requested ratio.

        Preferred mode:
            ratio length == number of chemical components, e.g. A1/A2

        Legacy mode:
            ratio length == number of primitive disorder configuration states
        """
        ratio = self.ratio

        if len(ratio) > 1:
            raise NotImplementedError(
                "ratio filtering not implemented for multiple per-assembly ratio arrays"
            )

        target_ratio = np.asarray(ratio[0], dtype=float)

        if target_ratio.sum() <= 0:
            raise ValueError(f"Target ratio sums to zero: {target_ratio}")

        target_ratio = target_ratio / target_ratio.sum()

        # New preferred mode: component-level composition, e.g. A1/A2.
        if hasattr(self, "component_labels") and len(target_ratio) == len(self.component_labels):
            primitive_indices = select_configs(
                config,
                supercell=self.supercell,
                maxgroups=self.ndisordergroups,
            )

            component_counts = Counter()

            for primitive_index in primitive_indices:
                component_counts.update(
                    self.primitive_config_metadata[int(primitive_index)]["components"]
                )

            values = np.asarray(
                [
                    component_counts.get(component, 0)
                    for component in self.component_labels
                ],
                dtype=float,
            )

            if values.sum() <= 0:
                return False

            computed_ratio = values / values.sum()

        # Legacy mode: ratio over primitive configuration indices.
        elif len(target_ratio) == self.ndisordergroups:
            computed_ratio = np.bincount(
                config,
                minlength=self.ndisordergroups,
            ).astype(float)

            computed_ratio = computed_ratio / computed_ratio.sum()

        else:
            raise ValueError(
                "Ratio length does not match orgdisord state space. "
                f"Got ratio length {len(target_ratio)}. "
                f"Component labels are {getattr(self, 'component_labels', None)}. "
                f"Primitive-state length is {self.ndisordergroups}."
            )

        return np.allclose(
            target_ratio,
            computed_ratio,
            atol=self.ratio_tolerance,
        )

    def get_supercell_config(self, config) -> Atoms:
        """
        Build a supercell from pre-aligned primitive molecular images.

        Do not atom-wise wrap primitive cells here. The primitive images already
        contain whole molecules in the intended periodic image.
        """
        na, nb, nc = self.supercell
        cell = np.asarray(self.cell)
        ndisordergroups = self.ndisordergroups

        transformation_matrix = np.diag([na, nb, nc])
        supercell_matrix = transformation_matrix @ cell

        supercell_atoms = Atoms(cell=supercell_matrix, pbc=True)

        for icell, c in enumerate(
            select_configs(
                config,
                supercell=self.supercell,
                maxgroups=ndisordergroups,
            )
        ):
            temp = self.images[int(c)].copy()

            # Do not call temp.wrap().
            # Do not call make_whole_molecule() here.
            # The primitive image should already be whole and correctly placed.
            temp.set_cell(cell)
            temp.set_pbc(True)

            ic = icell % nc
            ib = (icell // nc) % nb
            ia = icell // (nc * nb)

            R = np.asarray([ia, ib, ic], dtype=float) @ cell

            temp.translate(R)
            supercell_atoms.extend(temp)

        return supercell_atoms
    
    def _component_label(self, asslabel, group):
        """
        Return the chemical component label for a disorder group.

        This uses atom labels written by cspy-disord prepare-cif, e.g.

            A1_S001_C1   -> A1
            A2_S001_C1   -> A2
            A2m2_S001_C1 -> A2

        If this fails, fall back to the raw assembly/group label.
        """
        try:
            labels = group.atoms.get_array("labels")

            if len(labels) > 0:
                prefix = str(labels[0]).split("_")[0]

                match = re.match(r"(A\d+)", prefix)

                if match:
                    return match.group(1)

                return prefix

        except Exception:
            pass

        return f"{asslabel}{group.label}"

    def _component_counts_for_config(self, config, asslabel):
        """
        Count selected disorder groups for one primitive-cell configuration.

        This avoids using molecular formulae, so two disorder alternatives with
        the same formula remain distinguishable as e.g. A1 and A2.
        """
        assembly = self.disordered_structure.get_assembly(asslabel)
        disorder_groups = assembly.disorder_groups

        counts = Counter()

        if len(disorder_groups) == 1:
            label = self._component_label(asslabel, disorder_groups[0])
            counts[label] += len(config)
            return dict(counts)

        for c in config:
            group = disorder_groups[int(c)]
            label = self._component_label(asslabel, group)
            counts[label] += 1

        return dict(counts)

    def _raw_group_labels(self):
        """
        Return raw disorder group labels in CIF assembly/group order.

        Example:
            S001 group 1 -> S0011
            S001 group 2 -> S0012
            S002 group 1 -> S0021
        """
        labels = []

        for assembly in self.disordered_structure.disorder_assemblies:
            for group in assembly.disorder_groups:
                labels.append(
                    self._component_label(assembly.label, group)
                )

        return labels