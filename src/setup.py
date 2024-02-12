from setuptools import setup, find_packages

setup(
    name="NWShedEnv",
    version="1.0.0",
    author="Abhilash G",
    author_email="abhignavami@yahoo.com",
    description="An optimized OpenAi gym's environment to simulate the Traffic Scheduling problem. Adapted from JSS Env Pierre Tassel",
    url="https://github.com/Abhgit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "gym",
        "pandas",
        "numpy",
        "plotly",
        "imageio",
        "psutil",
        "requests",
        "kaleido",
        "pytest",
        "codecov",
    ],
    include_package_data=True,
    zip_safe=False,
)
