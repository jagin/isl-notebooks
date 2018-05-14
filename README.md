# isl-notebooks
This repository contains R and Python notebooks code for [An Introduction to Statistical Learning with Applications in R](http://www-bcf.usc.edu/%7Egareth/ISL/).

<img src='http://www-bcf.usc.edu/%7Egareth/ISL/ISL%20Cover%202.jpg' width=20%>

It's a learning playground for the book exercises including codes from other repositories (see [Credits](#credits)) in the form of dockerized Jupyter notebooks so it is easier to quickly run and experiment with.

## Notebooks

2. Statistical Learning  
    2.3 Lab - Introduction to [[R](http://nbviewer.jupyter.org/github/jagin/isl-notebooks/blob/master/notebooks/R/2%20Statistical%20Learning/2.3%20Lab%20-%20Introduction%20to%20R.ipynb)][[[Python](http://nbviewer.jupyter.org/github/jagin/isl-notebooks/blob/master/notebooks/Python/2%20Statistical%20Learning/2.3%20Lab%20-%20Introduction%20to%20Python.ipynb)]  
    2.4 Exercises - [[R](http://nbviewer.jupyter.org/github/jagin/isl-notebooks/blob/master/notebooks/R/2%20Statistical%20Learning/2.4%20Exercises.ipynb)]  

## Running Jupyter

With [Docker](https://www.docker.com/community-edition) you can quickly setup Jupyter environment to run the notebooks and do your own explorations.  
For more details see [opinionated stacks of ready-to-run Jupyter applications in Docker](https://github.com/jupyter/docker-stacks).

To run the isl-notebook container run the following command:

```
docker-compose up --build
```

Wait for:

```
...
isl-notebook    |     Copy/paste this URL into your browser when you connect for the first time,
isl-notebook    |     to login with a token:
isl-notebook    |         http://localhost:8888/?token=your_token
```

to be displayed on your console and follow the instruction.

If you need to run some additional commands in the container run:

```
bash -c clear && docker exec -it isl-notebook sh

```

### References

- G. James, D. Witten, T. Hastie, R. Tibshirani, [An Introduction to Statistical Learning with Applications in R](http://www-bcf.usc.edu/%7Egareth/ISL/), Springer Science+Business Media, 2013
- Hastie, T., Tibshirani, R., Friedman, J. (2009). [The Elements of Statistical Learning](http://statweb.stanford.edu/%7tibs/ElemStatLearn/), Second Edition, Springer Science+Business Media, 2009

### Resoures

- [Statistical Learning](https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about) by [Stanford University](http://www.stanford.edu/)
- [In-depth introduction to machine learning in 15 hours of expert videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/) by [Data School](http://www.dataschool.io)
- [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview)
- [Data Science and Machine Learning Bootcamp with R](https://www.udemy.com/data-science-and-machine-learning-bootcamp-with-r/learn/v4/overview)

### Credits

- [IntroStatLearning](https://github.com/ppaquay/IntroStatLearning)
- [ISL-python](https://github.com/emredjan/ISL-python)
- [ISLR-python](https://github.com/JWarmenhoven/ISLR-python)
- [stat-learning](https://github.com/asadoughi/stat-learning)