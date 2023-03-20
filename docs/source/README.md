<p align="center"> <img src="https://raw.githubusercontent.com/MannLabs/OmicLearn/master/OmicLearn/utils/OmicLearn_black.png" height="270" width="277" /> </p>
<h2 align="center">Online version: <a href="https://share.streamlit.io/MannLabs/OmicLearn/OmicLearn/OmicLearn.py" target="_blank">OmicLearn</a> </h2>

<h2 align="center"> 📰 Manual and Documentation: <a href="https://OmicLearn.readthedocs.io/en/latest/" target="_blank">OmicLearn ReadTheDocs </a> </h2>

![OmicLearn Tests](https://github.com/MannLabs/OmicLearn/workflows/OmicLearn%20Tests/badge.svg)
![OmicLearn Python Badges](https://img.shields.io/badge/Tested_with_Python-3.9-blue)
![OmicLearn Version](https://img.shields.io/badge/Release-v1.3-orange)
![OmicLearn Release](https://img.shields.io/badge/Release%20Date-July%202022-green)
![OmicLearn License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

---
# OmicLearn

Transparent exploration of machine learning for biomarker discovery from proteomics and omics data. This is a maintained fork from [OmicEra](https://github.com/OmicEra/OmicLearn).

## Quickstart
A three minute quickstart video to showcase OmicLearn can be found [here](https://youtu.be/VE9pj1G89io).

## Manuscript
📰 <a href="https://doi.org/10.1021/acs.jproteome.2c00473" target="_blank">Open-access article: **Transparent Exploration of Machine Learning for Biomarker Discovery from Proteomics and Omics Data**</a>

> **Citation:** <br>
> Torun, F. M., Virreira Winter, S., Doll, S., Riese, F. M., Vorobyev, A., Mueller-Reif, J. B., Geyer, P. E., & Strauss, M. T. (2022). <br>
> Transparent Exploration of Machine Learning for Biomarker Discovery from Proteomics and Omics Data. <br>
> Journal of Proteome Research. https://doi.org/10.1021/acs.jproteome.2c00473 <br>


## Online Access

🟢  <a href="https://share.streamlit.io/MannLabs/OmicLearn/OmicLearn/OmicLearn.py" target="_blank"> Streamlit share</a>

This is an online version hosted by streamlit using free cloud resources, which might have limited performance. Use the local installation to run OmicLearn on your own hardware.

## Local Installation

### One-click Installation

You can use the one-click installer to install OmicLearn as an application locally.
Click on one of the links below to download the latest release for:

[**Windows**](https://github.com/MannLabs/OmicLearn/releases/latest/download/omiclearn_gui_installer_windows.exe), [**macOS**](https://github.com/MannLabs/OmicLearn/releases/latest/download/omiclearn_gui_installer_macos.pkg), [**Linux**](https://github.com/MannLabs/OmicLearn/releases/latest/download/omiclearn_gui_installer_linux.deb)

For detailed installation instructions of the one-click installers refer to the [documentation](https://OmicLearn.readthedocs.io/en/latest/ONE_CLICK.html).

### Python Installation

- It is strongly recommended to install OmicLearn in its own environment using [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

  1. Redirect to the folder of choice and clone the repository: `git clone https://github.com/MannLabs/OmicLearn`
  2. Create a new environment for OmicLearn: `conda create --name OmicLearn python=3.9`
  3. Activate the environment with  `conda activate OmicLearn`
  4. Change to the OmicLearn directory with `cd OmicLearn` and install OmicLearn with `pip install .`

- After a successful installation, type the following command to run OmicLearn:

  `python -m OmicLearn`

 - After starting the streamlit server, the OmicLearn page should be automatically opened in your browser (Default link: [`http://localhost:8501`](http://localhost:8501)


## Getting Started with OmicLearn

The following image displays the main steps of OmicLearn:

![OmicLearn Workflow](workflow.png)

Detailed instructions on how to get started with OmicLearn can be found **[here.](https://OmicLearn.readthedocs.io/en/latest/USING.html)**

On this page, you can click on the titles listed in the *Table of Contents*, which contain instructions for each section.

## Contributing
All contributions are welcome. 👍

📰 To get started, please check out our **[`CONTRIBUTING`](https://github.com/MannLabs/OmicLearn/blob/master/CONTRIBUTING.md)** guidelines.

When contributing to **OmicLearn**, please **[open a new issue](https://github.com/MannLabs/OmicLearn/issues/new/choose)** to report the bug or discuss the changes you plan before sending a PR (pull request).

We appreciate community contributions to the repository.
