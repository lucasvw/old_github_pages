---
layout: post
title:  "Smoothing data with weighted least squares regression"
description : "Weighted least squares (WLS) is compares to ordinary least squared (OLS) and used to smooth (spectral) data."
date:   2017-06-15 12:00:00
categories: main
comments: true
---

## Introduction: Spectral data and OLS

As an (astro-) physicists, a lot of your time goes into the analysis of some kind of spectrum. Whether you are studying some star far away or taking measurements on some material in a laboratory, very often spectrums (spectra?) are involved and very often they are noisy. For those unfamiliar: a spectrum generally shows the intensity of some phenomena as a function of wavelength.

Normally, when confronted with such noisy data, the first thing to do is to smooth the data. When I was still studying physics, we mostly used [software][1] for that purpose. Smoothing was nothing more than a click of a button.

Recently, I came to understand that smoothing can be performed with the use of weighted least squares (WLS) regression. I remember WLS from studying econometrics, at that time it was only introduced as a way to deal with heteroskedasticity. Which is a fancy word for describing data where the variance of one variable is not uniform when regressed against another variable. Now I don't want to go too much into that area at the moment, but if you are interested just take a look at this [image][2] to understand the problem and [these note][3] on how to deal with it.

Instead I want to explain how WLS can be used for smoothing of data. Let's start with some noisy data:

![noise](/assets/images/data.png) 

Now before we go into the "weighted" part of WLS, I first want to explain how you actually use this data in a regression, since that took me a while to wrap my head around. Normally, when doing regression or applying any kind of other machine learning algorithm, it is quite clear what the features (input) and what the labels (output) are. However, spectral data normally comes as a CSV in which the header-row specifies the different wavelengths, and all the remaining rows denote a single spectrum, with a single intensity value for each wavelength. It took me a while to understand that from a regression perspective, the intensity is actually the *label* ($$y$$) and the *features* ($$X$$) are the individual wavelengths for which intensities are recorded.

Let's use this knowledge to [do an -unweighted- least squares regression really quick][4]. We know that ordinary least squares (OLS) minimizes the following objective function (it's kind of in the name, innit?) : 

{% raw %}
\begin{align}
	RSS_{OLS} = \sum_i (y_i - x_i^T \theta)^2 
\end{align}
{% endraw %}

Which leads to the normal equation:

{% raw %}
\begin{align}
	(X^TX)\theta &= X^Ty \\\\\
\end{align}
{% endraw %}

Which, solved for $$\theta$$, becomes:

{%raw%}
\begin{align}
	\theta &= (X^TX)^{-1}X^Ty \\\\\ 
\end{align}
{% endraw %}

So that the OLS estimate $$\hat{y}$$ for $$y$$ becomes:
{%raw%}
\begin{align}
	\hat{y} &= X\theta \\\\\\
	\hat{y} &= X(X^TX)^{-1}X^Ty
\end{align}
{% endraw %}

We create our $$X$$ (constant term and wavelengths) and $$y$$ (intensity values) as follows:
{% raw %}
\begin{align}
	X = \begin{bmatrix}
	1 \\ 1150 \\\\\\
	1 \\ 1151 \\\\\\
	1 \\ 1152 \\\\\\
	\vdots \\\\\\
	1 \\ 1599 
	\end{bmatrix}
	\hspace{1cm} 
	y = \begin{bmatrix}
	 0.693 \\\\\\
	 1.464 \\\\\\
	 2.068 \\\\\\
	 \vdots \\\\\\
	 0.382
	\end{bmatrix}
\end{align}
{% endraw %}

And find something which is -as expected- quite useless:

![noise](/assets/images/dataWithOLS.png)

## Weighted least squares regression for smoothing of data

The idea here, is that we are not going to fit a single line, but a lot of very small pieces of line. In fact, we will perform a single (weighted) linear regression for *each point* at which the function will be evaluated. This means that our model is no longer *parametric*. In other words, the model is no longer fully described by a set of parameters $$\theta$$, instead it remains dependant on our input data. Let's see how that works:

The objective to minimize in weighted linear regression model is given by:

{% raw %}
\begin{align}
	RSS_{WLS} = \sum_i w_i(y_i - x_i^T \theta)^2 
\end{align}
{% endraw %}

So the only difference with $$RSS_{OLS}$$ is the introduction of a weight $$w_i$$ which *weighs* the importance of each squared error term $$ (y_i - x_i^T \theta)^2 $$. When $$w_i$$ is small, the error term of observation $$i$$ will be pretty much ignored, when $$w_i$$ is large, the observation will have a large impact on the fit. What we will do, is make $$w_i$$ dependant on the distance of $$x_i$$ to the point $$x$$ where we want to evaluate our model:

{% raw %}
\begin{align}
	w_i = f_{\tau}(x) = \exp\Big(\frac{-(x - x_i)^2}{2\tau^2}\Big)
\end{align}
{% endraw %}

Therefore, when we evaluate the model at some point $$x$$ we can be sure, that only those values close to $$x$$ will be taken into consideration. The further away we go from $$x$$, the less pronounced are the effects on the estimate. 

The *speed* by which the effect of observations decreases with their distance to $$x$$, is controlled by the parameter $$\tau$$. This is a parameter we have to set. For now, we set this parameter to 5.

It can be shown, that WLS leads to:
{%raw%}
\begin{align}
	\hat{y} &= X(X^TWX)^{-1}X^TWy
\end{align}
{% endraw %}

Where $$W$$ is the (diagonal) weight matrix as such:

{% raw %}
\begin{align}
	W = \frac{1}{2}\begin{bmatrix}
	w_0 \\ 0 \\ \cdots \\ 0 \\\\\\
	0 \\ w_1 \\ \cdots \\ 0 \\\\\\
	\vdots \\ \\\\\\
	0 \\ 0 \\ \cdots \\ w_m
	\end{bmatrix}
\end{align}
{% endraw %}

[Putting it all together][5], this results in something one can actualy work with:

![wls](/assets/images/dataWithWLS.png)

Personally, I think it is quite fascinating that a minor modification of OLS can result in such a different outcome for such a different use-case. This shows me, that having a thorough understanding of the mathematics and foundations behind an algorithm or approach is so valuable. It allows to take these foundations and re-use it in a different way. Which leads to tailored-solutions for a problem at hand.

Next post, I will use this smoothed data to perform functional regression, in which an unknown part of a curve is predicted from training data.







[1]:http://www.originlab.com
[2]:https://upload.wikimedia.org/wikipedia/en/5/5d/Hsked_residual_compare.svg
[3]:http://www3.grips.ac.jp/~yamanota/Lecture_Note_10_GLS_WLS_FGLS.pdf 
[4]:https://github.com/lucasvw/spectral-data-processing/blob/63e4fb6dbe9e64ae5adaafbef3a8e5f5e90fcf99/1.py
[5]:https://github.com/lucasvw/spectral-data-processing/blob/63e4fb6dbe9e64ae5adaafbef3a8e5f5e90fcf99/2.py

