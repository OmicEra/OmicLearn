# import sphinx_material
project = "OmicLearn"
copyright = "2023, Furkan M. Torun, Maximilian T. Strauss"
author = "Furkan M. Torun, Maximilian T. Strauss"
release = "1.4"
extensions = ["myst_parser"]
templates_path = ["_templates"]
exclude_patterns = []
html_sidebars = {"**": ["globaltoc.html", "localtoc.html", "searchbox.html"]}
html_theme = "sphinx_material"
html_title = f"OmicLearn v{release}"
html_short_title = f"OmicLearn v{release}"
language = "en"
html_last_updated_fmt = ""
html_favicon = "images/OmicLearn.ico"
html_logo = "images/OmicLearn_white.png"
source_suffix = [".md", ".rst"]
master_doc = "globaltoc"
extensions.append("sphinx_material")

html_theme_options = {
    "base_url": "https://omiclearn.readthedocs.io/",
    "repo_url": "https://github.com/MannLabs/OmicLearn/",
    "repo_name": "OmicLearn",
    "nav_title": f"OmicLearn v{release} Docs",
    "html_minify": True,
    "css_minify": True,
    "color_primary": "red",
    "color_accent": "light-red",
    "globaltoc_depth": 1,
    "globaltoc_collapse": True,
    "globaltoc_includehidden": True,
    "version_dropdown": False,
    "logo_icon": "description",
    "nav_links": [
        {"href": "index.html", "title": "Home Page", "internal": False},
        {
            "href": "ONE_CLICK.html",
            "title": "One-Click Installation",
            "internal": False,
        },
        {"href": "USING.html", "title": "Using OmicLearn", "internal": False},
        {"href": "METHODS.html", "title": "Methods", "internal": False},
        {"href": "VERSION-HISTORY.html", "title": "Version History", "internal": False},
        {"href": "RECOMMENDATIONS.html", "title": "Recommendations", "internal": False},
    ],
}
