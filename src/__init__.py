"""LightDiffusion-Next source package.

Keep package initialization lightweight.

Historically this module eagerly imported large parts of the application,
which pulled the generation pipeline into unrelated imports such as tests
that only needed `src.FileManaging.ImageSaver`. A Docker Space only needs the
package namespace to resolve submodules; it does not need this eager import
fan-out.
"""
