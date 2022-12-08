import setuptools

setuptools.setup(
    name="cleanunet",
    version="0.0.2",
    author="Yehor Smoliakov",
    author_email="egorsmkv@gmail.com",
    description="cleanunet",
    long_description="cleanunet",
    long_description_content_type="text/markdown",
    url="https://github.com/egorsmkv/CleanUNet",
    packages=setuptools.find_namespace_packages(include=['cleanunet.*']),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
    ]
)
