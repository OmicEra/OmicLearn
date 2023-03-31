#!bash

# Initial cleanup
rm -rf dist
rm -rf build
FILE=OmicLearn.pkg
if test -f "$FILE"; then
  rm OmicLearn.pkg
fi
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n omiclearninstaller python=3.8 -y
conda activate omiclearninstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_macos_gui
pip install "../../dist/omiclearn-1.3.1-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.10
pyinstaller ../pyinstaller/omiclearn.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../omiclearn/data/*.fasta dist/omiclearn/data

# Wrapping the pyinstaller folder in a .pkg package
mkdir -p dist/omiclearn/Contents/Resources
cp ../logos/omiclearn_logo.icns dist/omiclearn/Contents/Resources
mv dist/omiclearn_gui dist/omiclearn/Contents/MacOS
cp Info.plist dist/omiclearn/Contents
cp omiclearn_terminal dist/omiclearn/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../logos/omiclearn_logo.png Resources/omiclearn_logo.png
chmod 777 scripts/*

pkgbuild --root dist/omiclearn --identifier de.mpg.biochem.omiclearn.app --version 0.3.0 --install-location /Applications/OmicLearn.app --scripts scripts OmicLearn.pkg
productbuild --distribution distribution.xml --resources Resources --package-path OmicLearn.pkg dist/omiclearn_gui_installer_macos.pkg
