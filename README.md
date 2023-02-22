![](engrenage.png "")

# Engrenage (the code formerly known as BabyGRChombo)

Engrenage is a spherically symmetric BSSN code designed for teaching Numerical Relativity (NR), which is the solution of the Einstein Equations of General Relativity (GR) using numerical methods. The code includes a scalar field (obeying the [Klein Gordon equation](https://en.wikipedia.org/wiki/Klein–Gordon_equation) for a minimally coupled spin 0 field) as the matter source of the metric curvature.
It currently includes two physical examples - a black hole and a real scalar boson star (or oscillaton).

Warning: This code was not designed to be a good example of optimised python usage or even of numerical relativity. The goal was to write a code where some non trivial physical examples could be studied and users could get an overview of the different parts of a numerical relativity code in action, without the optimisation and level of detail that exists in a typical research code like its parent [GRChombo](https://github.com/GRChombo/GRChombo).

For more information and class resources see the [wiki](https://github.com/GRChombo/engrenage/wiki).

(Engrenage is the French word for a system of gears. Understanding any code is very much like understanding how the parts of a mechanical system fit together. It is also very common when coding to be "pris dans l'engrenage".)

# Installation

1. [Fork and clone](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
   this repository.
2. Create a [Python environment](https://docs.python.org/3/tutorial/venv.html),
   e.g. in `./env`:

    ```sh
    python3 -m venv ./env
    ```

3. Install the Python requirements:

    ```sh
    # Activate the Python environment
    . ./env/bin/activate
    # Install the requirements
    pip install -r ./requirements.txt
    ```

4. Run the Jupyter notebook use either:

    ```sh
    jupyter-lab
    ```
    
    or

    ```sh
    jupyter notebook
    ```
    (depending on the style of interface that your prefer)
    

# Acknowledgements

This code is based on a private spherically adapted (but not spherically symmetric) code by Thomas Baumgarte, and the NRpy code of Zac Etienne, in particular the formalism used closely follows that described in the papers [Numerical relativity in spherical polar coordinates: Evolution calculations with the BSSN formulation](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.87.044026) and [SENR/NRPy+: Numerical relativity in singular curvilinear coordinate systems](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.064036).

This code has also benefitted from input from Nils Vu @nilsvu ("You don't use python environments? I don't even know where to start..."), Leo Stein @duetosymmetry ("Why wouldn't you use the existing numpy functions for that?") and bug spotting from Cristian Joana @cjoana and Cheng-Hsin Cheng @chcheng3 when this code debuted at the ICERM Numerical Relativity Community Summer School in August 2022, and David Sola Gil @David-Sola-Gil in the London LTCC course February 2023. Thanks also to Marcus Hatton @MarcusHatton for the addition of animation to the figures.

The main developer of engrenage is [Katy Clough](https://www.qmul.ac.uk/maths/profiles/katyclough.html), who is supported by a UK STFC Ernest Rutherford Fellowship ST/V003240/1.

