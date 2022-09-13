import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ilsmc-rivasiker",
    version="0.0.6",
    author="Iker Rivas-González",
    author_email="irg@birc.au.dk",
    description="A phylogenetically aware SMC to infer the evolutionary history of species",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rivasiker/ilsmc",
    project_urls={
        "Bug Tracker": "https://github.com/rivasiker/ilsmc/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy', 'pandas', 'scipy', 'multiprocessing'],
    include_package_data=True,
)
