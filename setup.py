import setuptools

long_description = ''
with open("README.md", "r") as fh:
    long_description = fh.read()
 
setuptools.setup(
    name="gimp_labeling_converter",
    version="2.0.0",
    author="Parham Nooralishahi",
    author_email="parham.nooralishahi@gmail.com",
    description="it provides a CLI tool that converts XCF file to a new format suitable for labeling. The tool also make the users able to add their own handlers for addling support to other types of outputs.",
    license = "BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords = "gimp data_labeling segmentation",
    url = "https://github.com/parham/gimp-labeling-converter", 
    packages=setuptools.find_packages(),
    package_data={'': ['*.json']},
    entry_points={
        "console_scripts": [
            "gimp_labeling_converter = gimp_labeling_converter.__main__:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Utilities",
        'Topic :: Software Development',
    ],
    install_requires=[
        'gimpformats',
        'Pillow',
        'scikit-image',
        'glob2',
        'torch',
        'torchvision'
    ],
    include_package_data=True,
    python_requires='>=3.7'
)
