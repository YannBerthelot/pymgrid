from setuptools import setup, find_packages

setup(
    name="pymgrid",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.1.1",
    python_requires=">=3.0",
    include_package_data=True,
    install_requires=[
        "requests",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "cufflinks",
        "gym",
    ],
)
