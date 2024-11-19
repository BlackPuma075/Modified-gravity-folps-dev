# **DESILIKE Tests for Cosmological Inference**

This repository contains useful scripts and environments for performing parameter inference from cosmological surveys, specifically using the **DESILike** and **CosmoPrimo** packages.

---

## **Overview**
This repository focuses on:
- Implementing and testing cosmological inference pipelines.
- Extending the **FOLPS** framework to Modified Gravity (FMG) scenarios.
- Providing ready-to-use environments and examples for researchers working with cosmological data.

---

## **Requirements**
To use the **FOLPS** scripts, you need to download the official **DESILike** and **CosmoPrimo** packages from their respective repositories:
- [DESILike Repository](https://github.com/cosmodesi/desilike)
- [CosmoPrimo Repository](https://github.com/cosmodesi/cosmoprimo)

---

## **Installation Instructions**

### 1. **Set up the environment**
To use the **FMG** scripts create the required environment, following these steps. By default, the environment will be named `fmg_env`, but you can rename it if needed.

```bash
conda env export > environment.yml
conda env create -f environment.yml
```
### 2. Install the modified DESILike and CosmoPrimo packages
Run the following commands to install the modified branches of DESILike and CosmoPrimo:
```bash
python -m pip install git+https://github.com/cosmodesi/cosmoprimo@dgbranch
python -m pip install git+https://github.com/cosmodesi/desilike@dbranch
```
---

## **Usage**
To run the Modified Gravity FOLPS (FMG) scripts:

1. Ensure the FMG.py file is in the same directory as the scripts you are running.
⚠️ Note: This is a temporary requirement and will be improved in future updates.
2. Follow the examples provided in the repository for your specific tests.

--- 

## **Future improvements**
-Automating the installation of FMG.py to eliminate manual steps.
-Adding more detailed examples and pre-configured tests.
-Improving compatibility with other cosmological tools.
