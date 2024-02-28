# MHD-Boussinesq-paper

This repository contains the source code utilized to generate simulations of the Magnetohydrodynamic (MHD) equations employing a Boussinesq conjecture to investigate the Rayleigh-Taylor instability relevant to the research paper:

- A. Piterskaya, W. Miloch, and M. Mortensen, “A global spectral-Galerkin investigation of the Rayleigh-Taylor instability in plasma using an MHD-Boussinesq model”

The model described in the paper has been implemented within the spectral Galerkin framework Shenfun (https://github.com/spectralDNS/shenfun), version 4.1.1.

To facilitate the conda installation process, kindly refer to the 'environment.yml' file, which contains a comprehensive list of dependencies required to establish a fully operational shenfun environment.

# Codespace

The code in this repository can be tested using a codespace. Press the green Code button and choose to create a codespace on main. A virtual machince will then be created with all the required software in environment.yml installed in a coda environment. To activate this do

    source activate ./venv

in the terminal of the codespace after the installation is finished. You may then run the program using

    python MHDBoussinesq.py
