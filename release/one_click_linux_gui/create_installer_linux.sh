#!bash

# Initial cleanup
rm -rf dist
rm -rf build
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n omiclearn_installer python=3.8 -y
conda activate omiclearn_installer

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_linux_gui
# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "../../dist/OmicLearn-1.3-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.10
pyinstaller ../pyinstaller/OmicLearn.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../OmicLearn/data/*.fasta dist/OmicLearn/data
# WARNING: this probably does not work!!!!

# Wrapping the pyinstaller folder in a .deb package
mkdir -p dist/omiclearn_gui_installer_linux/usr/local/bin
mv dist/OmicLearn dist/omiclearn_gui_installer_linux/usr/local/bin/OmicLearn
mkdir dist/omiclearn_gui_installer_linux/DEBIAN
cp control dist/omiclearn_gui_installer_linux/DEBIAN
dpkg-deb --build --root-owner-group dist/omiclearn_gui_installer_linux/
