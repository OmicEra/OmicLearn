#import sphinx_material
project = "OmicLearn"
copyright = "2023, Furkan M. Torun, Maximilian T. Strauss"
author = "Furkan M. Torun, Maximilian T. Strauss"
release = "1.4"
extensions = ["myst_parser"]
templates_path = ["_templates"]
exclude_patterns = []
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}
html_theme = 'sphinx_material'
html_title = f"OmicLearn v{release}"
html_short_title = f"OmicLearn v{release}"
language = 'en'
html_last_updated_fmt = ''
html_favicon = 'images/OmicLearn.ico'
html_logo = 'images/OmicLearn_white.png'
source_suffix = ['.md', ".rst"]
master_doc = 'globaltoc'
extensions.append('sphinx_material')

html_theme_options = {
    'base_url': 'https://omiclearn.readthedocs.io/',
    'repo_url': 'https://github.com/MannLabs/OmicLearn/',
    'repo_name': 'OmicLearn',
    'html_minify': True,
    'css_minify': True,
    'color_primary': 'red',
    'color_accent': 'light-red',
    'globaltoc_depth': 1,
    'globaltoc_collapse': True,
    'globaltoc_includehidden': True,
    "version_dropdown": False,
    'nav_links': [], # {"href": "METHODS.html", "title": "METHODS", "internal": False}
}

