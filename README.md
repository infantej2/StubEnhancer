# StubEnhancer
### Quick Links
[Introduction](#Introduction)  
&nbsp;&nbsp;&nbsp;&nbsp;[Target Audience](#Target-Audience)  
&nbsp;&nbsp;&nbsp;&nbsp;[Implementation Details](#Implementation-Details)  
&nbsp;&nbsp;&nbsp;&nbsp;[Limitations](#Limitations)  
[Quick Start](#Running-StubEnhancer-Locally)  
[Hosting](#Hosting)  
&nbsp;&nbsp;&nbsp;&nbsp;[Docker](#Docker)  
&nbsp;&nbsp;&nbsp;&nbsp;[Podman](#Podman)  
[Contribute Code](#Contribute-Code)

------
## Introduction

The objective of our application is to help our target audience make informed education and career decisions. We do this through interactive visualizations
that abstracts the relevant data in meaningful ways. Further, we offer our users access to predictive models that compliment our data.

#### Target Audience
As of right now, our target audience is limited to those residing within Alberta. Primarily, high school students who are deciding what career or education
paths to take. Additionaly, anyone within Alberta looking to inform their career and education decisions. We plan on expanding our application to all of 
Canada in the near future.

#### Implementation Details
Data processing was done using the **Pandas** library and other related statistical libraries that are associated with Python. The interactive visualizations were
built using **Plotly and Dash**. Machine learning for salary prediction was done using **Keras**, which is built ontop of the popular **TensorFlow** library. Other notable libraries used in this projects implementaiton were **Numpy, matplotlib, scipy, and sklearn**.

#### Limitations
Our project has several limitations associated with it in its current state:
1. Our data is limited to Alberta salary data from the years 2005-2014.
2. We assume that the *Years After Graduation* column in our data set maps 1:1 with years experience.
3. We assume that the salaries are for individuals working in the fields they have degrees in, however, in reality not everyone who gets a degree works in their field.
4. Our machine learning model has an RMSE of approximately $11,000. An ideal RMSE would be magnitudes lower. However, it currently predicts along the trend line that we would expect. So, although not entirely accurate, it is useful to users.

------
## Running StubEnhancer Locally

Run the following command in your terminal: ```pip install -r requirements.txt```
**Note that not all Python versions (such as 3.8.0) seem to support tensorflow. If tensorflow will not install via pip, consider upgrading python...**
*And you're ready!* Load up your choice of IDE and run app.py

------
## Hosting

If you would like to host an instance of StubEnhancer, you can do so directly by running app.py with Python, after installing all requirements with the command ```pip install -r requirements.txt```
StubEnhancer utilizes port 8050 (by default). You may forward this port directly, or use a reverse-proxy to redirect requests to and from StubEnhancer.

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
