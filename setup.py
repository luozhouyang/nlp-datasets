import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlp_datasets",
    version="1.2.0",
    description="A dataset utils repository based on tf.data. For tensorflow 2.x only!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luozhouyang/nlp-datasets",
    author="ZhouYang Luo",
    author_email="zhouyang.luo@gmail.com",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[

    ],
    license="MIT License",
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
