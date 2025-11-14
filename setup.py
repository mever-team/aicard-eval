from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="aicard-eval",
    version='0.1.0',
    packages=find_packages(),
    install_requires=["datasets",
                    "numpy",
                    "validators",
                    "scikit-learn",
                    "scikit-image",
                    "jiwer",
                    "psutil",
                      ],
    description="Evaluation module for aicard.",
    author="CERTH",
    author_email="gnikoul@gmail.com",
    url="https://github.com/mever-team/aicard-eval",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Development Status :: 2 - Pre-Alpha",
    ],
    python_requires=">=3.11",
    long_description=long_description,
    long_description_content_type="text/markdown"
)
