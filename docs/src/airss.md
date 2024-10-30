# Ab initio Random Structure Searching (AIRSS)

Ab initio random structure searching (AIRSS)[^1][^2] is an approach to
search for low energy structure by simply generating random structures,
followed by local optimisation. Local optimisation is a relatively
simple and mathematically well-established, it found a nearby local
minimum in the configuration space by simply going down-hill in energy.
The potential energy surface (PES) can be divided into many
basins-of-attraction. Performing a local optimisation inside a basin
would lead us to bottom of this basin, e.g. the local minimum.

From first glance, such random sampling approach is deemed to fail for
complex and high-dimensional configuration space due to the existence of
many high energy local minima, making the chance of landing in the basin
of attraction of the global minimum diminishing. However, one must not
forget that the potential energy surface (PES) is a physical quantity,
and hence their structure follow certain rules. While the number of high
energy local minima may be overwhelming, the chance of landing into a
basins-of-attraction is proportional to its hypervolume, and the lower
the energy, the larger the hypervolume, favouring even a random sampler.
In addition, we know that a large portion of the configuration space is
not likely to contain any low energy minima - the atoms the bonds
between them do have well-known finite sizes. The region of the PES
including atoms very close to each other are unlikely to have any low
energy minima, and hence can be excluded from sampling. Likewise,
structures with atoms far apart from each other should also be excluded
from sampling. With a few physical constraints, such as the species-wise
minimum separations, including symmetry operations, a simple random
search approach can be made highly efficient for locating the ground
state structure and other low energy polymorphs. While the structures
are *randomly* generated, the parameters controlling the generation
process are *not randomly chosen* - they are motivated by physical
reasons.

The phrase *ab initio* not only means that this approach **can** be used
with first-principles methods for energy evaluation, but also that it
**should** be used with them, since it is the physical nature of the PES
that is been exploited here. While random search can be applied with
parameterized interatomic potentials, the latter are unlikely to
reproduce the true PES especially in the region far away from fitted
phases. Hence, it is not a surprise if random search does not work well
in those cases.

A consequence of random sampling is that the computational workload can
be made fully parallel and distributed. There is no need to coordinate
the search as a whole, each worker can work independently little or no
inter-communication at all. In constrast, *global optimisation*
methods such as basin hopping, requires each calculation to be performed
in serial. Population based approaches such as genetic algorithms and
particle swarm optimisation allow certain degree of parallelism within a
single generation (tenth of structures), but each generation still have
to be evaluated iteratively.

This means that for random searching:

-   The underlying DFT calculations can be parallelised over only a few
    CPUs each, maximising the parallel efficiently which otherwise can
    drop sharply with increasing core counts.
-   The elimination of iterative process means there is no dilemma of
    *exploration or exploitation*.
-   A further consequence is that the DFT calculations can be performed
    at relatively low basis set qulaity and accuracy to maximise the
    speed, usually at several times lower in the cost compared to normal
    calculations.
-   In addition, most structures include and perserv symmetry operations
    throughout the process, which can be used to speed-up DFT
    calculations by several folds.

# Ingredients of a search

This section gives a brief introduction of how to run a search using
`airss` along.

The searching process is rather simple: a random structure is generated
and subsequently relaxed. However, both steps needs to be optimised for
efficient searching.

## Structure generation

In particular, the random structure should be not generated purely
\"randomly\", but instead should based on a set of pre-determined
constraints, to ensure that the output structures are physically sound.
If not done so, the search can be rather inefficient[^1][^2][^3] (often
incorrectly interpreted as baseline).

The structure generation process is done by the `buildcell` program,
which takes a CASTEP `cell` file style input. The key quantities pass
for providing the constraints are:

-   Estimated volume
-   Species-wise minimum separations
-   Number of symmetry operations to include

The question is: how can one know this prior to even performing the
search? One simple way is to perform a short search with some guessed
values. Such search may not find the true ground state, but the low
energy structure would be able to provide the *cues* of what a ground
state structure may look like. The first two parameters can thereby be
estimated from the low energy structure of such \"pilot\" search.
Alternatively, if there are known experimental structure, the first two
parameters can be estimated from those as well.

!!! tip
    `cat <xxx>.res | cryan -g` is your friend for exacting these parameters.
    If the input structure is not in the SHELX format, the `cabal` tool can
    be used to convert it into that!

If one looks at typically polymorphs of a compounds, they do not vary
too much (say \< 20%) in terms of density and bond lengths. The
species-wise minimum separation may be inferred from that of chemically
similar materials. The exact values of the first two parameters would
not make a huge difference in terms of search efficiency as long as they
are sensible. In fact, one may want to use minimum separation drawn
randomly from a certain

Finally, one may want to include symmetry operations in the generated
structures - experimental structures are rarely *P1* after all.
Typically, two to four symmetry operations are included in the generated
structure. The actual structure may gain more symmetry during the
geometry operatimisation process, so the it is no necessary to have
chosen specific space group in the first place.


!!! note
    A side issue of imposing symmetry operations is that once a space group
    is chosen, the multiplicity of a atom will not change during subsequent
    relaxations. For example, atoms at the general positions will not be
    able to move to special positions. The default rule in `airss` is to
    maximise the occupation of general positions, which reduces the overall
    degrees of freedom. This also seems to follow the trend in most known
    crystals, but there can be exceptions. More special positions can be
    occupied by specifying the `#ADJGEN` setting. Mind that the general
    positions of a low symmetry space group can still become special
    positions of a higher symmetry one or that of a smaller unit cell.

The template `cell` file generate by `gencell` contains other default
settings which will not be detailed here.

## DFT relaxation

Each generated structure needs to be relaxed to a local minimium by some
means. First-principles based methods are preferable as they provide
realistic potential energy surfaces that are relatively smooth.
Typically, this is done by CASTEP[^4] , although other plane wave DFT
code may also be used as well. CASTEP is preferred because it is well
tested for this kind of tasks. It has robust electronic and ionic
minimisation routines and soft pseudopotentials (QC5) optimised for
throughput. The self-adaptive parallelisation in CASTEP also make it
easy to deploy calculations on computing resources, since no manual
input of parallelisation scheme is needed.

The typical setting is to use the QC5 pseudopotentials that requires
very low cut off energies (300 - 340 eV) in order to maximise the speed.
These potentials are probably not accurate enough for regular
calculations, but they are sufficient for sampling the potential energy
surfaces. The depth and the relative positions the local minima may be
slightly wrong, but using them would be allow us to local these low
energy structure much faster. Since there is no ranking taking place
during the search to direct the sampling region (e.g. unlike GA or PSO),
it is not necessary to obtain accurate energy and structures at this
stage. In the end, a set of high quality calculations (typically using
the C19 pseudopotential set) needs be applied to refine the results and
obtain reliable energy orderings. This process is applied to only a
subset of low energy structures that are already near the local minimum
needs to be processed.

# When to (not) stop

In crystal structure prediction, there is no way to make sure the ground
state structure found is indeed true ground state, unless one performs a
exhaustive sampling of the potential energy surface, which is
impractical for any high-dimensional space. However, for a single
search, ones still have to have a stopping criteria.

# The **AIRSS** package

The **AIRSS** package is a open-source collection of tools and scripts for
performing *ab initio* random structure searching (AIRSS) (which
confusinly has the same name), and analysing the results. The key
components of this package includes:

## `buildcell` 

The main work horse for generating structures. This file reads a
input *seed* file from stdin and outputs generated structuer in
stdout. Both input and outputs files are in the CASTEP\'s `cell`
format, and the former contains special directives on how the random
Structure should be generated.

It is often useful to generate a initial cell using the
`gencell` utility with `gencell <volume> <units> [<specie> <number>]`.
For example to search for `SrTiO3`, one can input:

``` console
gencell 60 4 Sr 1 Ti 1 O 3
```

This prepares a seed to search for four formula units of `SrTiO3`, each
formula unit is expected to have a volume of 60 $\mathrm{Å^3}$.


The content of the generated file `4SrTiO3.cell` is shown below:

```
%BLOCK LATTICE_CART
3.914865 0 0
0 3.914865 0
0 0 3.914865
%ENDBLOCK LATTICE_CART

#VARVOL=60

%BLOCK POSITIONS_FRAC
Sr 0.0 0.0 0.0 # Sr1 % NUM=1
Sr 0.0 0.0 0.0 # Sr2 % NUM=1
Sr 0.0 0.0 0.0 # Sr3 % NUM=1
Sr 0.0 0.0 0.0 # Sr4 % NUM=1
Ti 0.0 0.0 0.0 # Ti1 % NUM=1
Ti 0.0 0.0 0.0 # Ti2 % NUM=1
Ti 0.0 0.0 0.0 # Ti3 % NUM=1
Ti 0.0 0.0 0.0 # Ti4 % NUM=1
O 0.0 0.0 0.0 # O1 % NUM=1
O 0.0 0.0 0.0 # O2 % NUM=1
O 0.0 0.0 0.0 # O3 % NUM=1
O 0.0 0.0 0.0 # O4 % NUM=1
O 0.0 0.0 0.0 # O5 % NUM=1
O 0.0 0.0 0.0 # O6 % NUM=1
O 0.0 0.0 0.0 # O7 % NUM=1
O 0.0 0.0 0.0 # O8 % NUM=1
O 0.0 0.0 0.0 # O9 % NUM=1
O 0.0 0.0 0.0 # O10 % NUM=1
O 0.0 0.0 0.0 # O11 % NUM=1
O 0.0 0.0 0.0 # O12 % NUM=1
%ENDBLOCK POSITIONS_FRAC

##SPECIES=Sr,Ti,O
##NATOM=3-9
##FOCUS=3

#SYMMOPS=2-4
##SGRANK=20
#NFORM=1
##ADJGEN=0-1
#SLACK=0.25
#OVERLAP=0.1
#MINSEP=1-3 AUTO
#COMPACT
#CELLADAPT
##SYSTEM={Rhom,Tric,Mono,Cubi,Hexa,Orth,Tetra}

KPOINTS_MP_SPACING 0.07

SYMMETRY_GENERATE
SNAP_TO_SYMMETRY

%BLOCK SPECIES_POT
QC5
%ENDBLOCK SPECIES_POT

%BLOCK EXTERNAL_PRESSURE
0 0 0
0 0
0
%ENDBLOCK EXTERNAL_PRESSURE
```

As you can see, the input file for `buildcell` is essentially an marked
up `cell` file for CASTEP. Lines starting with `#` provides directives
for `buildcell`, although those with `##` will still be ignored.

## Options for `buildcell`

Several key tags are explained below:

- `VARVOL`

    Defines the *variable* volume of the unit cell, the exact volume
    will be randomise at a harded coded range of ± 5%. The actual unit
    cell vectors will be randomised, and those in the `LATTICE_CART`
    block takes no effect.

- `POSITIONS_ABC`

    Defines the *initial* positions of the atoms in the unit cell. The
    syntax after the `#` is `<set name> % [<tags>...]`. Where
    `<set name>` can be used to define rigid fragments by setting the
    same value for all of the linked sites. Since each atom should be
    considered independently in this example, each line has a different
    value. After the `%` comes the site-specific settings. The `NUM`
    allows a single site to be multiplicated. For example, adding 12
    oxygen atoms can achieved specified by `O 0. 0. 0. # O % NUM=12`
    instead of doing it line by line.

- `SYMOPS`

    Defines the number of symmetry operations to be included in the
    randomised structure. The `-` defines the range and a random value
    will be drawn from.

- `SGRANK`

    This tags can be enabled to bias the (randomly chosen) space group
    towards those that are more frequently found in the ICSD. The value
    defines a threshold rank for accepting the space group.

- `NFORM`

    This tags defines the number of formula units to be included in the
    generated structure. Here, the cell has the formula $\ce{Sr4Ti4O12}$
    according to the `POSITIONS_ABC` block. If one changes `NFORM` to be
    `2`, then the effective chemical formula would be $\ce{Sr8Ti8O24}$.
    If `NCORM` is not defined, the composition will be affect by
    `SYMOPS`, placing all atoms in the general positions.

- `ADJGEN`

    This tags is used to modify the number of general positions when
    generating symmetry-containing structures. By default, the number of
    general positions is maximised. This tags allows more special
    postions to be occupied if possible. For blind search, there is no
    need to use this tag in most cases.

- `MINSEP`

    This tags defines the specie-wise minimum separations and is one of
    the few tags that need to be manually changed by hand. The default
    `1-3 AUTO` means to randomly set minimum separations *per
    specie-pair* between 1 Å and 3 Å, but also try to extract and use
    that of the best structure if possible. The latter part is achieved
    by looking for a `.minsep` file in the current wroking directory,
    which is generated by the `airss.pl` script on-the-fly. This
    approach cannot be used by `DIPS` searches each calculation will be
    run from different directories (and probably also different
    machines). Initial values may be composed by the knowledge of common
    bond lengths. The `cryan -g` command can also be used to extract
    `MINSEP` from existing structures.

- `SLACK,OVERLAP`

    These two tags controls the process of modifying the structure to
    satisfy the minimum separations. Roughly speaking, a geometry
    optimisation is performed ,each species-pair having a hard shell
    potential. The tag `SLACK` defines how *soft* the hard shell
    potential constructed will be, and `OVERLAP` defines the threshold
    for acceptance. There is usually no need to vary these two tags.

- `COMPACT,CELLADAPT`

    Controls if the cell should be compacted and deformed based on the
    hard shell potentials. There is usually no need to change these two
    tags.

- `SYSTEM`

    Allows limiting the generated structure to certain crystal systems,
    which can be useful to bias the search based on prior knowledge.

## Options for CASTEP

Lines not marked up by `#` are often passed through, below are
descriptions of some CASTEP-related tags that goes in the `cell` file.


!!! note
    The `buildcell` program will not always pass through the native CASTEP keys, so check
    the output cell carefully.

- `KPOINTS_MP_SPACING`

    Defines kpoints spacing in unit of $\mathrm{2\pi\,Å^{-1}}$. Usually
    a not so well-converged setting can be used for the initial search.

- `SYMMETRY_GENERATE`

    Let CASTEP determine the symmetry for acclerating calculations. You
    will almost always want to have this.

- `SNAP_TO_SYMMETRY`

    Snap the atoms to the positions according to the determined
    symmetry. You will almost always want to have this.

- `EXTERNAL_PRESSURE`

    The upper triangle of the external stress tenser. It does not get
    passed through after `buildcell`.

- `HUBBARD_U`

    The Hubbard-U values for each specie and orbital. For example,
    `Mn d:3` will apply an effective U of 3 eV to the d orbital of Mn.

# The `param` file

The `param` file read by CASTEP only, and not used for building structure.
A default `param` file for CASTEP from `gencell` looks like this:

```
task                 : geometryoptimization
xc_functional        : PBE
spin_polarized       : false
fix_occupancy        : false
metals_method        : dm
mixing_scheme        : pulay
max_scf_cycles       : 1000
cut_off_energy       : 340 eV
opt_strategy         : speed
page_wvfns           : 0
num_dump_cycles      : 0
backup_interval      : 0
geom_method          : LBFGS
geom_max_iter        : 20
mix_history_length   : 20
finite_basis_corr    : 0
fixed_npw            : true
write_cell_structure : true
write_checkpoint     : none
write_bib            : false
write_otfg           : false
write_cst_esp        : false
write_bands          : false
write_geom           : false
bs_write_eigenvalues : false
calculate_stress     : true

```

Several tags that you may want to modify are:

- `xc_functional`

    Defines the exchange-correlation functional to be used. Since PBE
    often bias towards less bonded (larger volume) phases, one may want
    to use PBEsol instead.

- `spin_polarized`

    If set to `true` the calculation will be spin polarized. Note that
    CASTEP does not have default set for each site, so one have to break
    the symmetry manually in addition to setting this tag.

- `fixed_npw`

    aka, fix the number of plane waves. CASTEP default to constant
    basis-quality during variable-cell geometry geometry optmisation
    rather than the constant basis set in many other plane wave codes.
    While this allows getting a reliable final energy with restarts, the
    `castep_relax` script already handles automatically restart anyway.
    Having consistant basis-quality may not be optimum when large pulay
    stress is present, e.g. using less well-converged cut off energies.
    Hence, it is preferable to set it to `true` and use the constant
    basis set approach.

- `cut_off_energy`

    The cut off energy should be sufficient for the pseudopotential
    used. There is no need to high quality but harder pseudopotentials
    in the initial search in most case.

- `geom_method`

    Algorithm for geometry optmisation. `LBFGS` works well cases,
    otherwise `tpsd` may be used instead.

- `mixing_scheme`

    This tags controls the charge density mixer. The `pulay` mixer is OK
    for most case, otherwise one can use `broyden` instead if
    convergence struggles.

- `max_scf_cycles`

    Keep a large value to avoid CASTEP giving up the structure due to
    electronic convergence instability.

- `opt_strategy`

    Use `speed` to keep everything in the RAM and avoid any slow disk
    IO.

- `geom_max_iter`

    Number of geometry optimisation per run. Since `castep_relax` will
    repetively restart calculation, use a small value so the basis set
    is reset every so often. For fixed cell optimisation, one can use a
    large value in combination with apporiate `castep_relax` arguments
    and avoid restarts.

- `metals_method`

    Use `dm` (density mixing) for speed.


## `airss.pl`

The main driver script for performing the search. It read command
line arguments and performs random structure generation and runs DFT
calculations in serial, and stop until the specified number of
structure has been generated. Because the search is embarrsingly
parallel, one can just launch as many `airss.pl` as they like to
occupy all computational resources. For example, to sample 800
structure using 128 cores, one can launch 8 `airss.pl` script each
using 16 cores and sampling 100 structures. The result of `airss.pl`
are saved in the SHELX format with suffix `res`. These files
contains both the crystal structure and the calculated quantities.
While DISP does not use this script directly, it is recommanded
that the user familiarise themselves with operating AIRSS using
it.

## `cryan`

A tool to analyse the relaxed structures. It can be used to rank
structures according to energy as well as eliminating nearly
identical structures and extracting species-wise minimum distances.
It also has many other advanced features as such decomposing a
structure into modular units.

## `cabal`

A tool for convert different file formats. It is used internally by
various scripts. One very useful feature is to convert file into
SHELX format so they can be processed by `cryan`.

## `castep_relax`

This is script supplied by the `airss` package. The main idea is to
restart the DFT calculations every certain number of geometry step, in
order to update the basis set when running in the *fixed basis set* mode
with variable unit cell. By default, the script does several restarts
with very small number of ionic steps, as it is expected that the volume
could change significantly in the very begining. Afterwards, the
`geom_max_iter` setting in the `param` is respected. The optimisation is
terminated if CASTEP reports that the optimization is successful in two
consecutive restats, of if the maximum number of steps has been reached.

The input syntax is:

```
castep_relax <maxit> <exe> <sim> <symm> <seed>
```

These arguments are explained as below:

-   `<maxit>`: The maximum total number of geometry optimisation steps
    among all restarts. The optimisation is *assumed* to be finished if
    this number of steps has been reached.
-   `<exe>`: The launch command for running CASTEP, including the MPI
    launch command and its arguments.
-   `<sim>`: Aswitch to enable check if the same structure has been
    found before. It should be disabled by setting it to `0` in DISP, as
    each calculation is run under a different directory.
-   If the `<symm>` switch is set to `1` the structure will be
    symmetrised on-the-fly during the restarts. It is typically turned
    off.
-   `<seed>`: The name of the seed for CASTEP.


[^1]: Pickard, C. J.; Needs, R. J. Ab Initio Random Structure Searching.
    Journal of physics. Condensed matter : an Institute of Physics
    journal 2011, 23 (5), 053201--053201.
    <https://doi.org/10.1088/0953-8984/23/5/053201>.

[^2]: Section 4.2.2, <https://doi.org/10.17863/CAM.55681>

[^3]: Figure 7, <https://doi.org/10.1016/j.cpc.2020.107810>

[^4]: Academic license for CASTEP can be obtained free of charge, see
    <http://www.castep.org>.

