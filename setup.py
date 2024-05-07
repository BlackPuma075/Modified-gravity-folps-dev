import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DESILIKE_Tests",
    version="0.0.1",
    author="Diego Gonzalez",
    author_email="",
    description="Internal functions for DESILIKE project tests.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BlackPuma075/DESILIKE_Tests",
    project_urls={
        "Bug Tracker": "https://github.com/BlackPuma075/DESILIKE_Tests/issues"
    },
    license="BSD 3-Clause",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

