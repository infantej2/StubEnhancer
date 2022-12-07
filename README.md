# StubEnhancer
### Table of Contents
[Quick Start](#Quick-Start:-Running-StubEnhancer-Locally)
[Hosting](#Hosting)
    [Docker](#Docker)
    [Podman](#Podman)
[Contribute Code](#Contribute-Code)

------
## Introduction


------
## Quick Start: Running StubEnhancer Locally

Run the following command in your terminal: ```pip install -r requirements.txt```
**Note that not all Python versions (such as 3.8.0) seem to support tensorflow. If tensorflow will not install via pip, consider upgrading python...**
*And you're ready!* Load up your choice of IDE and run app.py

------
## Hosting

If you would like to host an instance of StubEnhancer, you can do so directly by running app.py with Python, after installing all requirements with the command ```pip install -r requirements.txt```
StubEnhancer utilizes port 8050 (by default). You may forward this port directly, or use a reverse-proxy to redirect requests to StubEnhancer.

Alternatively, a [Dockerfile](/Dockerfile) is provided within the solution folder. With this file, you can use either Docker or Podman in the following ways:

#### Docker
Build the container (each time you want to pull updates, you can also rebuild using the same command):
```docker build --no-cache -t stub_enhancer_container .```
**Note that pip may appear frozen for a few seconds during build, simply let it run.**

Run the container:
```docker run -d -p 0.0.0.0:8050:8050 --name stub_enhancer stub_enhancer_container```
**Note that you can switch which port you want to access the container from by replacing "-p 0.0.0.0:8050:8050". For instance, "-p 0.0.0.0:80:8050" will allow you to access the container via port 80.**
*Consult [Docker's documentation](https://docs.docker.com/engine/reference/run/) if you need to modify or add changes for your situation.*

#### Podman
Build the container:
```podman build --no-cache -t stub_enhancer_container .```
**Note that pip may appear frozen for a few seconds during build, simply let it run.**

Run the container:
```podman run -d -p 0.0.0.0:8050:8050 --name stub_enhancer stub_enhancer_container```
**Note that you can switch which port you want to access the container from by replacing "-p 0.0.0.0:8050:8050". For instance, "-p 0.0.0.0:80:8050" will allow you to access the container via port 80.**
*Consult [Podman's documentation](https://docs.podman.io/en/latest/markdown/podman-run.1.html) if you need to modify or add changes for your situation.*

------
## Contribute Code
After January 1st 2023, we will be accepting code contributions! Feel free to submit a pull request after this date if you would like to contribute to the project.