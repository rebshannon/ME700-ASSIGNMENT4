# ME700-ASSIGNMENT4

# Set Up

Set up adapted from https://github.com/Lejeune-Lab-Graduate-Course-Materials/fenicsX  

FEniCSX reference:  https://fenicsproject.org/download/

Set up the conda environment and test that the code is functioning. 

1. The conda environment can be setup on an SCC terminal. Start by loading miniconda.
  
    ```bash 
    module load miniconda
    ```
2. Create and activate the environment.  
    ```bash
    mamba create -n fenicsx-env
    ```
    ```bash
    mamba activate fenicsx-env
    ```
3. Install necessary pacakges with mamba.  
    ```bash
    mamba install -c conda-forge fenics-dolfinx mpich pyvista
    ```  
4. Install necessary pacakges with pip.
    ```bash
    pip install imageio
    pip install gmsh
    pip install PyYAML
    ```
5. On VSCode, activate the environment you just created.  
     ```bash
    mamba activate fenicsx-env
    ```
6. To run an example walkthrough in FEniCSX, open *PoiseuilleFlow_walkthrough.ipynb*.
