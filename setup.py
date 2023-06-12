import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trails-rivasiker",
    version="0.0.1",
    author="Iker Rivas-González",
    author_email="irg@birc.au.dk",
    description="Tree reconstruction of ancestry using incomplete lineage sorting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rivasiker/trails",
    project_urls={
        "Bug Tracker": "https://github.com/rivasiker/trails/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_namespace_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy', 'pandas', 'scipy', 'ray', 'numba', 'biopython'],
    include_package_data=True,
)
