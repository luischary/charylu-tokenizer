from setuptools import find_packages, setup

setup(
    name="charylu-tokenizer",
    packages=find_packages(include=["charylutokenizer"]),
    version="0.0.1",
    description="Biblioteca com tokenizadores criados por Luis Chary",
    author="Luis Felipe Chary",
    install_requires=["tokenizers==0.19.1", "numpy==2.0.0"],
    tests_require=["pytest==8.2.2"],
    test_suite="tests",
)
