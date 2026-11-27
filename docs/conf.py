# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "LLaMEA"
copyright = "2024-2026, Niki van Stein"
author = "Niki van Stein"

try:
    from importlib.metadata import version as _version

    release = _version("llamea")
except Exception:  # pragma: no cover - docs built without the package
    release = ""
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinxext.opengraph",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google-style docstrings (Args:, Returns:, Raises:)
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

# The notebooks need an LLM; show them as saved instead of running them.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
napoleon_use_ivar = True
autodoc_member_order = "bysource"
autodoc_mock_imports = ["xgboost", "lmstudio", "mlx_lm", "torch", "transformers"]

# CHANGELOG.md starts at "##" once its title line is skipped.
suppress_warnings = ["myst.header"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "media/banner"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"LLaMEA {release}".strip()
html_favicon = "_static/llamea-mark.png"
html_css_files = ["css/llamea.css"]

# LLaMEA shares BLADE's design (Furo, layout, cards, dark canvas) with a
# teal/mint accent instead of BLADE's violet.
html_theme_options = {
    "light_logo": "llamea-mark.png",
    "dark_logo": "llamea-mark-dark.png",
    "source_repository": "https://github.com/XAI-liacs/LLaMEA",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#13806f",
        "color-brand-content": "#13806f",
        "color-admonition-title-background--note": "#e3f6f0",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8ee6c0",
        "color-brand-content": "#8ee6c0",
        "color-background-primary": "#0f1117",
        "color-background-secondary": "#151821",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/XAI-liacs/LLaMEA",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}

# Link previews (social media, chat).
ogp_site_url = "https://xai-liacs.github.io/LLaMEA/"
ogp_image = "https://xai-liacs.github.io/LLaMEA/_static/llamea-social-preview.png"
ogp_description_length = 200
