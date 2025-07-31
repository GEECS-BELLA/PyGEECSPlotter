from setuptools import setup

setup(
    name="PyGEECSPlotter",
    version="0.0.3",
    description="GEECS Plotter Library",
    author="Alex Picksley",
    packages=["PyGEECSPlotter"],
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "ipywidgets",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "opencv-python",
        "pypng"
    ],
)
