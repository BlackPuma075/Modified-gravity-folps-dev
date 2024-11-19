## DESILIKE TESTS FOR COSMOLOGICAL INFERENCE
This is a repository where you can find some useful scripts and environments to make parameter inference from cosmological surveys, specifically using the desilike and cosmoprimo packages. 
To use the FOLPS scripts you only need to download desilike and cosmoprimo packages as it is in the original repository in https://github.com/cosmodesi/desilike and https://github.com/cosmodesi/cosmoprimo 
To run the Modified gravity FOLPS (FMG) you can recreate the environment using the next steps:
```bash
conda env export > environment.yml
conda env create -f environment.yml
