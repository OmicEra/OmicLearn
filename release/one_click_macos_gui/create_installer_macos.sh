#!bash

# Initial cleanup
rm -rf dist
rm -rf build
FILE=omiclearn.pkg
if test -f "$FILE"; then
  rm omiclearn.pkg
fi
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n omiclearninstaller python=3.10 -y
conda activate omiclearninstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_macos_gui
pip install "../../dist/OmicLearn-1.4-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.10
pyinstaller ../pyinstaller/omiclearn.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../omiclearn/data/*.fasta dist/omiclearn/data

# Wrapping the pyinstaller folder in a .pkg package
mkdir -p dist/omiclearn/Contents/Resources
cp ../logos/omiclearn_logo.icns dist/omiclearn/Contents/Resources
cp ../logos/omiclearn_logo.ico dist/omiclearn/Contents/Resources/omiclearn.ico
cp ../logos/omiclearn_logo.ico dist/omiclearn/Contents/Resources/omiclearn_logo.ico
mv dist/omiclearn_gui dist/omiclearn/Contents/MacOS
cp Info.plist dist/omiclearn/Contents
cp omiclearn_terminal dist/omiclearn/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../logos/omiclearn_logo.png Resources/omiclearn_logo.png
cp ../logos/omiclearn_logo.ico Resources/omiclearn_logo.ico
cp ../logos/omiclearn_logo.ico Resources/omiclearn.ico
chmod 777 scripts/*

pkgbuild --root dist/omiclearn --identifier de.mpg.biochem.omiclearn.app --version 0.3.0 --install-location /Applications/omiclearn.app --scripts scripts omiclearn.pkg
productbuild --distribution distribution.xml --resources Resources --package-path omiclearn.pkg dist/omiclearn_gui_installer_macos.pkg
