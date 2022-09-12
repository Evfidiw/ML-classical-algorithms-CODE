## 0x00 pre

- The main references: 
    
    - Andrew Ng's cs229
- Recommended prerequisite knowledge：
    - Python for ML, and some library functions such as Numpy, Pandas, Matplotlib （Just need to understand, we will be proficient in use)
    - some mathematical knowledge such as Calculus, Linear algebra, Probability theory
    - ~~we don't need "H He Li Be B"~~

- what's this note?

    note the venation of courses in mathematics 

## 0x01 introduction

> A computer program is said to learn from experience E with respect to  some class of tasks T and performance measure P, if its performance at  tasks in T, as measured by P, improves with experience E.

## 0x02 linear model

- thought: make use of "2 dim to n dim"

- how to expend "h"
    $$
    h(x)=\theta_0+\theta_1X\\
    h(x)=\sum{^m_{j=0}\theta_jX_j}\\
    \theta=\left[
    \begin{array}{l}
    \theta_0\\
    \theta_1\\
    ...\\
    \theta_m
    \end{array}
    \right],
    X=\left[
    \begin{array}{l}
    X_0\\
    X_1\\
    ...\\
    X_m
    \end{array}
    \right]\\
    h(x)=\theta^TX
    $$

### linear regression

- how to get "θ": (2 method)

    - gradient descent 

        - cost function
            $$
            J(\theta)=min\{\sum{^m_{j=0}(h_\theta(x^{(i)})-y^{(i)})^2}\}
            $$

            why choose "least squares"? 
            $$
            y^{(i)}=\theta X^{(i)}+\epsilon^{(i)}\\
            \epsilon^{(i)}are~distributed~IID,because~of~central~limit~theorm,\epsilon^{(i)}\sim N(0,\sigma)\\
            L(\theta)=\prod{^n_{i=1}}p(y^{(i)}|x^{(i)},\theta)=\prod{^n_{i=1}}\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})\\
            l(\theta)=logL(\theta)=nlog\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\frac{1}{2}\sum(y^{(i)}-\theta^Tx^{(i)})^2\\
            let~likelihood~max,minimise~\frac{1}{2}\sum(y^{(i)}-\theta^Tx^{(i)})^2
            $$

        - Least Mean Square

        $$
        \theta_j:=\theta_j+\alpha\frac{\partial}{\partial\theta_j}J(\theta_j)\\
        
        \frac{\partial}{\partial\theta_j}J(\theta_j)\\
        \begin{align}
        &=\frac{\partial}{\partial\theta_j}(\sum{^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2})\\
        &=\sum(h_\theta(x)-y)\frac{\partial}{\partial\theta_j}(\sum{^m_{j=0}\theta_jx_j}-y_j)\\
        &=(h_\theta(x)-y)x_j
        \end{align}\\
        
        \theta_j:=\theta_j+\alpha(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j
        $$

        - gradient descent:

            - Batch Gradient Descent

            $$
            \theta_j:=\theta_j+\alpha\sum(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j
            $$

            - Stochastic Gradient Descent

    - normal equation

$$
\nabla_\theta J(\theta)=\left[
\begin{array}{l}
\frac{\partial J}{\partial\theta_0}\\
...\\
\frac{\partial J}{\partial\theta_m}
\end{array}
\right]\\

f:R^{n\times n}\rightarrow R\\
\theta=\left[
\begin{array}{l}
\theta_0\\
\theta_1\\
...\\
\theta_m
\end{array}
\right],
X=\left[
\begin{array}{l}
X_0\\
X_1\\
...\\
X_m
\end{array}
\right]\\
J(\theta)=\frac{1}{2}\sum(h(x^{(i)})-y^{(i)})^2=\frac{1}{2}(X\theta-y)^T(X\theta-y)\\

\nabla_\theta J(\theta_j)\\
\begin{align}
&=\frac{1}{2}\nabla_\theta(\theta^TX^T-y^T)(\theta X-y)\\
&=\frac{1}{2}∇_θ(θ^T(X^TX)θ−y^T(Xθ)−y^T(Xθ))\\
&=X^TX\theta-X^Ty\\
&=0
\end{align}\\

thus,\ X^TXθ=X^Ty\\
\theta=(X^TX)^{−1}X^Ty.
$$

- Locally weighted linear regression
    - a non-parametric algorithm

    - cost function:

    $$
    J(\theta)=\sum{^m_{i=1}}w^{(i)}(\theta^TX^{i}-y^{(i)})^2\\
    or,J(\theta)=W(Y-X^T\theta)^T(Y-X^T\theta)\\
    w^{(i)}=exp(\frac{-(x^{(i)}-x^{i})^2}{2\tau^2})\\
    if|x^{(i)}-x|is\ small,w^{(i)}closer\ to\ 1\\
    if|x^{(i)}-x|is\ large,w^{(i)}closer\ to\ 0\\
    $$

    - normal equation
        $$
         w=(X^TWX)^{−1}X^T W y
        $$
        

- generalized linear model

### Logistic regression

- logistic/logistic function function
    $$
    h_\theta(x)=g(\theta^TX)=\frac{1}{1+e^{-\theta X}}\\
    g(z)=\frac{1}{1+e^{-z}}
    $$

- LMS get "θ"
    $$
    P(y=1| x; θ) = h_θ(x)\\
    P(y=0| x; θ) = 1−h_θ(x)\\
    L(\theta)=p(y|X;\theta)=\prod{^n_{i=1}}(h_θ(x^{(i)}))^{y(i)}(1−h_θ(x^{(i)})^{1−y(i)}\\
    l(\theta)=logL(\theta)=\sum{^n_{i=1}}y^{(i)}logh(x^{(i)})+(1−y^{(i)})log(1 − h(x^{(i)}))\\
    
    \frac{∂}{∂θ_j}l(θ)\\
    \begin{align}
    &=(y\frac{1}{g(θ^Tx)}−(1−y)\frac{1}{1−g(θ^Tx)})\frac{∂}{∂θ_j}g(θ^T x)\\
    &=(y\frac{1}{g(θ^Tx)}−(1−y)\frac{1}{1−g(θ^Tx)})g(\theta^Tx)(1-g(\theta^Tx))\frac{\partial}{\partial\theta_j}\theta^Tx\\
    &=(y-h_\theta(x))x_j
    \end{align}\\
    
    θ_j:=θ_j+α(y^{(i)}−h_θ(x^{(i)}))x{^{(i)}_j}
    $$

- Newton's method get "θ"
    $$
    θ:=θ−\frac{f(θ)}{f′(θ)}\\
    n~dim~θ:=θ−H^{−1}∇_θl(θ)\\
    Hessian~matrix:H_{ij}=\frac{\partial^2l(\theta)}{\partial\theta_i\partial\theta_j}
    $$

### linear Discriminant Analysis

- 

## 0x04 the perceptron

- The exponential family
    $$
    p(y;η) = b(y)exp(η^TT(y)−a(η))\\
    
    eg~Bernoulli~distribution:\\
    \begin{align}
    p(y; φ)&=φ^y(1−φ)^{1−y}\\
    &=exp(y logφ + (1 − y) log(1 − φ))\\
    &=exp((log(\frac{φ}{1 − φ}))y + log(1 − φ))
    \end{align}\\
    $$

- construct GLMs

## 0x05 Gaussian discriminant analysis

- generative learning algorithm 

- The multivariate normal distribution
    $$
    p(x; μ,Σ)=\frac{1}{(2π)d/2|Σ|1/2}exp(−\frac{1}{2}(x − μ)^TΣ^{−1}(x − μ))\\
    a~mean~vector~\mu\in\R^d,a~covariance~matrix~Σ\in\R^{d\times d}\\
    u=E[x]=\int_xxp(x;u,Σ)dx=\mu
    $$

- model

## 0x06 naive bayes classifier 

- feature vector x

- model
    $$
    x\in\{0,1\}^d\\
    p(x_1, . . . , x_{50000}|y)=\prod{^d_{j=1}}p(x_j|y)\\
    L(φ_y,φ_{j|y=0},φ_{j|y=1})=\prod{^n_{i=1}}p(x^{(i)},y^{(i)})
    $$
    

## 0x07 supports  vector machine



## 0x08 decision tree

