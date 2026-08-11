[1] Title: FAN: Fourier Analysis Networks
[1] URL Source: https://arxiv.org/abs/2410.02675
[1] Description: Abstract page for arXiv paper 2410.02675: FAN: Fourier Analysis Networks
[1] Published Time: Tue, 28 Oct 2025 01:08:41 GMT

[View PDF](https://arxiv.org/pdf/2410.02675)[HTML (experimental)](https://arxiv.org/html/2410.02675v6)

> Abstract:Despite the remarkable successes of general-purpose neural networks, such as MLPs and Transformers, we find that they exhibit notable shortcomings in modeling and reasoning about periodic phenomena, achieving only marginal performance within the training domain and failing to generalize effectively to out-of-domain (OOD) scenarios. Periodicity is ubiquitous throughout nature and science. Therefore, neural networks should be equipped with the essential ability to model and handle periodicity. In this work, we propose FAN, a novel neural network that effectively addresses periodicity modeling challenges while offering broad applicability similar to MLP with fewer parameters and FLOPs. Periodicity is naturally integrated into FAN's structure and computational processes by introducing the Fourier Principle. Unlike existing Fourier-based networks, which possess particular periodicity modeling abilities but face challenges in scaling to deeper networks and are typically designed for specific tasks, our approach overcomes this challenge to enable scaling to large-scale models and maintains general-purpose modeling capability. Through extensive experiments, we demonstrate the superiority of FAN in periodicity modeling tasks and the effectiveness and generalizability of FAN across a range of real-world tasks. Moreover, we reveal that compared to existing Fourier-based networks, FAN accommodates both periodicity modeling and general-purpose modeling well.

Comments:Accepted to NeurIPS'25
Subjects:Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
Cite as:[arXiv:2410.02675](https://arxiv.org/abs/2410.02675) [cs.LG]
(or [arXiv:2410.02675v6](https://arxiv.org/abs/2410.02675v6) [cs.LG] for this version)
[https://doi.org/10.48550/arXiv.2410.02675](https://doi.org/10.48550/arXiv.2410.02675)

arXiv-issued DOI via DataCite

## Submission history

From: Yihong Dong [[view email](https://arxiv.org/show-email/d48bc246/2410.02675)] 

**[[v1]](https://arxiv.org/abs/2410.02675v1)** Thu, 3 Oct 2024 17:02:21 UTC (5,171 KB)

**[[v2]](https://arxiv.org/abs/2410.02675v2)** Sat, 9 Nov 2024 19:07:44 UTC (5,172 KB)

**[[v3]](https://arxiv.org/abs/2410.02675v3)** Fri, 31 Jan 2025 17:00:03 UTC (5,704 KB)

**[[v4]](https://arxiv.org/abs/2410.02675v4)** Wed, 2 Apr 2025 04:15:51 UTC (5,649 KB)

**[[v5]](https://arxiv.org/abs/2410.02675v5)** Tue, 30 Sep 2025 02:19:28 UTC (5,307 KB)

**[v6]** Sun, 26 Oct 2025 19:15:11 UTC (5,333 KB)

[2] Title: FAN: Fourier Analysis Networks
[2] URL Source: https://arxiv.org/html/2410.02675v6
[2] Description: 
[2] Published Time: Tue, 28 Oct 2025 01:08:45 GMT

Yihong Dong 1, Ge Li 1 1 1 footnotemark: 1, Yongding Tao 1, Xue Jiang 1, Kechi Zhang 1, Jia Li ♂1, 

Jinliang Deng 2, Jing Su 3, Jun Zhang 3, Jingjing Xu 3

1 School of Computer Science, Peking University 

2 The Hong Kong University of Science and Technology 3 ByteDance 

dongyh@stu.pku.edu.cn, lige@pku.edu.cn

###### Abstract

Despite the remarkable successes of general-purpose neural networks, such as MLPs and Transformers, we find that they exhibit notable shortcomings in modeling and reasoning about periodic phenomena, achieving only marginal performance within the training domain and failing to generalize effectively to out-of-domain (OOD) scenarios. Periodicity is ubiquitous throughout nature and science. Therefore, neural networks should be equipped with the essential ability to model and handle periodicity. In this work, we propose FAN, a novel neural network that effectively addresses periodicity modeling challenges while offering broad applicability similar to MLP with fewer parameters and FLOPs. Periodicity is naturally integrated into FAN’s structure and computational processes by introducing the Fourier Principle. Unlike existing Fourier-based networks, which possess particular periodicity modeling abilities but face challenges in scaling to deeper networks and are typically designed for specific tasks, our approach overcomes this challenge to enable scaling to large-scale models and maintains general-purpose modeling capability. Through extensive experiments, we demonstrate the superiority of FAN in periodicity modeling tasks and the effectiveness and generalizability of FAN across a range of real-world tasks. Moreover, we reveal that compared to existing Fourier-based networks, FAN accommodates both periodicity modeling and general-purpose modeling well. ††footnotetext: This work was supported by a cooperation project between Peking University and ByteDance Company. During this time, Yihong was also an intern at ByteDance.‡‡footnotetext: The code is available at [https://github.com/YihongDong/FAN](https://github.com/YihongDong/FAN)

## 1 Introduction

The flourishing of modern machine learning and artificial intelligence is inextricably linked to the revolutionary advancements in the foundational architecture of general-purpose neural networks. For instance, multi-layer perceptron (MLP)(Rosenblatt, [1958](https://arxiv.org/html/2410.02675v6#bib.bib1); Haykin, [1998](https://arxiv.org/html/2410.02675v6#bib.bib2)) plays a pivotal role in laying the groundwork for current deep learning models, with its expressive power guaranteed by the universal approximation theorem(Hornik et al., [1989](https://arxiv.org/html/2410.02675v6#bib.bib3)). Recent claims about the impressive performance of large models on various tasks are typically supported by Transformer architecture(Vaswani et al., [2017](https://arxiv.org/html/2410.02675v6#bib.bib4); Touvron et al., [2023](https://arxiv.org/html/2410.02675v6#bib.bib5); OpenAI, [2023](https://arxiv.org/html/2410.02675v6#bib.bib6)). In this context, the community’s enthusiasm for research on neural networks has never diminished. Some emerged neural networks demonstrate notable capabilities in specific fields(Gu and Dao, [2023](https://arxiv.org/html/2410.02675v6#bib.bib7); Liu et al., [2024](https://arxiv.org/html/2410.02675v6#bib.bib8)), sparking widespread discussion within the community.

Beneath the surface of apparent prosperity, we uncover a critical issue that remains in existing general-purpose neural networks: they struggle to model the periodicity from data, especially in OOD scenarios. We showcase this issue through an empirical study as illustrated in Figure [1](https://arxiv.org/html/2410.02675v6#S1.F1 "Figure 1 ‣ 1 Introduction ‣ FAN: Fourier Analysis Networks"). The results indicate that existing neural networks, including MLP (Rosenblatt, [1958](https://arxiv.org/html/2410.02675v6#bib.bib1)), KAN (Liu et al., [2024](https://arxiv.org/html/2410.02675v6#bib.bib8)), and Transformer (Vaswani et al., [2017](https://arxiv.org/html/2410.02675v6#bib.bib4)), face difficulties in fitting periodic functions, even on a simple sine function. Although they demonstrate some proficiency in interpolation within the domain of training data, they tend to falter when faced with extrapolation challenges of test data. This signifies that their generalization capacity is primarily dictated by the scale and diversity of the training data, rather than by the learned principles of periodicity to perform reasoning.

Periodicity is an essential characteristic in various forms of reasoning and generalization, as it provides a basis for predictability in many natural and engineered systems by leveraging recurring patterns in observations. Besides periodic phenomena, non-periodic phenomena can also be contextualized or explained within some larger or more macro-periodic framework. Although some Fourier-based networks exhibit particular periodic modeling abilities, they are primarily tailored for specific tasks (Silvescu, [1999](https://arxiv.org/html/2410.02675v6#bib.bib9); Liu, [2013](https://arxiv.org/html/2410.02675v6#bib.bib10)) and do not work well as the networks deepen (Liu et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib11)), which limits their applicability to the general task such as language modeling (Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib12)). However, our goal is to exploit periodicity to benefit a broader range of tasks including language modeling. To achieve this, we aim to develop a neural network that accommodates modeling and reasoning capabilities for periodicity while maintaining general-purpose modeling capability.

![Image 1: Refer to caption](https://arxiv.org/html/2410.02675v6/x1.png)

Figure 1: The performance of different neural networks within and outside the domain of their training data for the sine function, where x is a scalar variable. 

In this paper, we propose Fourier Analysis Network (FAN), a novel neural network built upon the principle of Fourier Analysis. By leveraging the power of Fourier Series, we enable the neural network to model periodic patterns and extrapolate beyond them, offering the network a way to model the general principles from the data. FAN follows two core principles, the first ensures that its periodic modeling capacity scales with network depth, while the second guarantees periodic modeling is available throughout the network. These principles allow it to scale to deeper networks, a capability where existing Fourier neural networks fall short. As a result, FAN exhibits exceptional capabilities in periodicity modeling, while maintaining broad applicability to the general task, which holds great potential as a substitute for MLP, with fewer parameters and FLOPs.

To verify the effectiveness of FAN, we conduct extensive experiments from three main aspects: 1) For periodicity modeling, FAN achieves significant improvements in fitting both basic and complex periodic functions, compared to existing neural networks (including MLP, KAN, and Transformer), particularly in OOD scenarios. 2) FAN shows superior performance in various real-world tasks, such as symbolic formula representation, time series forecasting, and language modeling. Using FAN outperforms the representative models in various tasks, including MLP, KAN, LSTM, Mamba, and Transformer. 3) Compared to existing Fourier-based networks, FAN accommodates both periodicity modeling and general-purpose modeling well. The advantageous characteristics and promising results indicate that FAN has the potential to become a basic component for building fundamental large models.

## 2 Preliminary Knowledge

Fourier Analysis(Stein and Weiss, [1971](https://arxiv.org/html/2410.02675v6#bib.bib13); Duoandikoetxea, [2024](https://arxiv.org/html/2410.02675v6#bib.bib14)) is a mathematical framework that decomposes functions into their constituent frequencies, revealing the underlying periodic structures within complex functions. At the heart of this analysis lies Fourier Series(Tolstov, [2012](https://arxiv.org/html/2410.02675v6#bib.bib15)), which expresses a periodic function as an infinite sum of sine and cosine terms. Mathematically, for a function f(x), its Fourier Series expansion can be represented as:

f(x)=a_{0}+\sum_{n=1}^{\infty}\left(a_{n}\cos\left(\frac{2\pi nx}{T}\right)+b_{n}\sin\left(\frac{2\pi nx}{T}\right)\right),(1)

where T is the period of the function, and the coefficients a_{n} and b_{n} are determined by integrating the function over one period:

a_{n}=\frac{1}{T}\int_{0}^{T}f(x)\cos\left(\frac{2\pi nx}{T}\right)\,dx,\quad b_{n}=\frac{1}{T}\int_{0}^{T}f(x)\sin\left(\frac{2\pi nx}{T}\right)\,dx.(2)

The power of Fourier Series lies in its ability to represent a wide variety of functions, including non-periodic functions through periodic extensions, enabling the extraction of frequency components. Building on this math foundation, FAN aims to embed the periodic characteristics into network architecture, enhancing generalization capabilities and performance on various tasks, particularly in scenarios requiring the identification of patterns and regularities.

![Image 2: Refer to caption](https://arxiv.org/html/2410.02675v6/x2.png)

Figure 2: Illustrations of FAN layer \phi(x) vs. MLP layer \Phi(x).

## 3 Fourier Analysis Network (FAN)

In this section, we first construct a naive neural network modeled by the formula of Fourier Series. Then, by modifying and improving it, we design FAN adhering to two core principles. Finally, we discuss the difference between the FAN layer and MLP layer.

Consider a task involving input-output pairs \{x_{i},y_{i}\}, with the objective of identifying a function f(x):\mathbb{R}^{d_{x}}\rightarrow\mathbb{R}^{d_{y}} that approximates the relationship such that y_{i}\approx f(x_{i}) for all x_{i}, where d_{x} and d_{y} denote the dimensions of x and y, respectively. We first construct a shallow neural network f_{\text{S}}(x) that represents Fourier Series expansion of the function, specifically \mathcal{F}\{f(x)\}, as described in Eq. ([1](https://arxiv.org/html/2410.02675v6#S2.E1 "Equation 1 ‣ 2 Preliminary Knowledge ‣ FAN: Fourier Analysis Networks")), we can express f_{\text{S}}(x) as follows:

\displaystyle f_{\text{S}}(x)\displaystyle\triangleq a_{0}+\sum_{n=1}^{N}\left(a_{n}\cos\left(\frac{2\pi nx}{T}\right)+b_{n}\sin\left(\frac{2\pi nx}{T}\right)\right),(3)
\displaystyle\mathop{=}\limits^{(\text{I})}a_{0}+\sum_{n=1}^{N}\left(w^{c}_{n}\cos\left(w^{\text{in}}_{n}x\right)+w^{s}_{n}\sin\left(w^{\text{in}}_{n}x\right)\right),
\displaystyle\mathop{=}\limits^{(\text{II})}B+[w^{c}_{1},w^{c}_{2},\cdots,w^{c}_{n}]\cos([w^{\text{in}}_{1}||w^{\text{in}}_{2}||\cdots||w^{\text{in}}_{n}]x)
\displaystyle\quad+[w^{s}_{1},w^{s}_{2},\cdots,w^{s}_{n}]\sin([w^{\text{in}}_{1}||w^{\text{in}}_{2}||\cdots||w^{\text{in}}_{n}]x)
\displaystyle=B+W_{c}\cos(W_{\text{in}}x)+W_{s}\sin(W_{\text{in}}x),
\displaystyle\mathop{=}\limits^{(\text{III})}B+W_{\text{out}}[\cos(W_{\text{in}}x)||\sin(W_{\text{in}}x)],

where B\in\mathbb{R}^{d_{y}},W_{\text{in}}\in\mathbb{R}^{N\times d_{x}}, and W_{\text{out}}\in\mathbb{R}^{d_{y}\times 2N} are learnable parameters, (\text{I}) follows that the computation of a_{n} and b_{n} computed via Eq. ([2](https://arxiv.org/html/2410.02675v6#S2.E2 "Equation 2 ‣ 2 Preliminary Knowledge ‣ FAN: Fourier Analysis Networks")) is definite integral, (\text{II}) and (\text{III}) follows the equivalence of the matrix operations, [\cdot||\cdot] and [\cdot,\cdot] denotes the concatenation along the first and second dimension, respectively.

To fully leverage the advantages of deep learning, we can stack the aforementioned network f_{\text{S}}(x) to form a deep network f_{\text{D}}(x), where the i-th layer, denoted as l_{i}(x), retains the same structural design as f_{\text{S}}(x). Therefore, f_{\text{D}}(x) can be formulated as:

f_{\text{D}}(x)=l_{L}\circ l_{L-1}\circ\cdots\circ l_{1}\circ x,(4)

where l_{1}\circ x denotes the application of the left function l_{1} to the right input x, that is l_{1}(x). However, we discover that the direct stacking of f_{\text{S}}(x) results in the primary parameters of the network f_{\text{D}}(x) focusing on learning the angular frequency (\omega_{n}=\frac{2\pi n}{T}), thereby neglecting the learning of the Fourier coefficients (a_{n} and b_{n}), as follows:

\displaystyle f_{\text{D}}(x)=l_{L}(l_{L-1}\circ l_{L-2}\circ\cdots\circ l_{1}\circ x)(5)
\displaystyle=B^{L}+W^{L}_{\text{out}}[\cos(W^{L}_{\text{in}}(l_{1:L}\circ x)||\sin(W^{L}_{\text{in}}(l_{1:L}\circ x))]

where l_{1:L}\circ x is defined as l_{L-1}\circ l_{L-2}\circ\cdots\circ l_{1}\circ x, W^{L}_{\text{in}}(l_{1:L}\circ x) is used to approximate the angular frequencies, and W^{L}_{\text{out}} is used to approximate the Fourier coefficients. We can find that the capacity of f_{\text{D}}(x) to fit the Fourier coefficients is independent of the depth of f_{\text{D}}(x), which is an undesirable outcome. It will limit the network’s representation ability, hindering to address the complex tasks.

Table 1: Comparison of FAN layer and MLP layer, where d_{\text{p}} is a hyperparameter of FAN layer and defaults to \frac{1}{4}d_{\text{output}} in this paper, d_{\text{input}} and d_{\text{output}} denote the input and output dimensions of the neural network layer, respectively. In our evaluation, the floating point of operations (FLOPs) for any arithmetic operations are considered as 1, and for Boolean operations as 0.

MLP Layer FAN layer
Formula\Phi(x)=\sigma(B_{m}+W_{m}x)\phi(x)=[\cos(W_{p}x)||\sin(W_{p}x)||\sigma(B_{\bar{p}}+W_{\bar{p}}x)]
Num of Params(d_{\text{input}}\times d_{\text{output}})+d_{\text{output}}(1-\frac{d_{p}}{d_{\text{output}}})\times((d_{\text{input}}\times d_{\text{output}})+d_{\text{output}})
FLOPs 2\times(d_{\text{input}}\times d_{\text{output}}) +\text{FLOPs}_{\text{non-linear}}\times d_{\text{output}}(1-\frac{d_{p}}{d_{\text{output}}})\times 2\times(d_{\text{input}}\times d_{\text{output}}) +\text{FLOPs}_{\text{non-linear}}\times d_{\text{output}}

To this end, we design FAN based on the following principles: 1) the capacity of FAN to represent the Fourier coefficients should be positively correlated to its depth; 2) the output of any hidden layer can be employed to model periodicity using Fourier Series through the subsequent layers. The first one enhances the expressive power of FAN for periodicity modeling by leveraging its depth, while the second one ensures that the features of FAN’s intermediate layers are available to perform periodicity modeling.

Suppose we decouple f_{\text{S}}(x) as follows:

f_{\text{S}}(x)=f_{out}\circ f_{in}\circ x,(6)

where

\displaystyle f_{in}(x)=[\cos(W_{\text{in}}x)||\sin(W_{\text{in}}x)],(7)
\displaystyle f_{out}(x)=B+W_{\text{out}}x.(8)

To satisfy both principles, the inputs of the intermediate layers in FAN necessitate to employ f_{in} and f_{out} simultaneously, rather than applying them sequentially.

Finally, FAN is designed on this basis, with the FAN layer \phi(x) defined as below:

\phi(x)\triangleq[\cos(W_{p}x)||\sin(W_{p}x)||\sigma(B_{\bar{p}}+W_{\bar{p}}x)],(9)

where W_{p}\in\mathbb{R}^{d_{x}\times d_{p}},W_{\bar{p}}\in\mathbb{R}^{d_{x}\times d_{\bar{p}}}, and B_{\bar{p}}\in\mathbb{R}^{d_{\bar{p}}} are learnable parameters (with the hyperparameters d_{p} and d_{\bar{p}} indicating the first dimension of W_{p} and W_{\bar{p}}, respectively), the layer output \phi(x)\in\mathbb{R}^{2d_{p}+d_{\bar{p}}}, and \sigma denotes the activation function. Under this definition, the MLP layer can be regarded as a special form of Eq. ([9](https://arxiv.org/html/2410.02675v6#S3.E9 "Equation 9 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks")), when W_{p} are learned to be zero metrics, which provides a way for FAN to maintain general-purpose modeling abilities as MLP.

![Image 3: Refer to caption](https://arxiv.org/html/2410.02675v6/x3.png)

![Image 4: Refer to caption](https://arxiv.org/html/2410.02675v6/x4.png)

Figure 3: The performance of FAN in periodicity modeling compared to MLP, KAN, and Transformer (Part I), where the green line represents the test data within the domain of training data, while the blue line represents the test data outside the domain of training data.

The entire FAN is defined as the stacking of the FAN layer \phi(x) as follows:

\text{FAN}(x)=\phi_{L}\circ\phi_{L-1}\circ\cdots\circ\phi_{1}\circ x,(10)

where

\small{\phi_{l}(x)=\left\{\begin{array}[]{ll}[\cos(W^{l}_{p}x)||\sin(W^{l}_{p}x)||\sigma(B^{l}_{\bar{p}}+W^{l}_{\bar{p}}x)],&\text{if }l<L,\\
B^{L}+W^{L}x,&\text{if }l=L,\end{array}\right.}(11)

#### The difference between FAN and MLP.

The illustrations of FAN layer \phi(x) vs. MLP layer \Phi(x) are shown in Figure [2](https://arxiv.org/html/2410.02675v6#S2.F2 "Figure 2 ‣ 2 Preliminary Knowledge ‣ FAN: Fourier Analysis Networks"). Note that the FAN layer \phi(x) computed via Eq. ([9](https://arxiv.org/html/2410.02675v6#S3.E9 "Equation 9 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks")) can seamlessly replace the MLP layer \Phi(x) computed via Eq. ([12](https://arxiv.org/html/2410.02675v6#A1.E12 "Equation 12 ‣ Appendix A MLP ‣ FAN: Fourier Analysis Networks")) in various models with fewer parameters and FLOPs, achieved by sharing the parameters and computation of Sin and Cos parts. The number of parameters and FLOPs of the FAN layer compared to the MLP layer are presented in Table [1](https://arxiv.org/html/2410.02675v6#S3.T1 "Table 1 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks"). The reduction ratio of parameters and FLOPs is about \frac{d_{p}}{d_{\text{output}}}, which is set to \frac{1}{4} by default in this paper.

## 4 Experiments

In this section, we first verify the superiority of FAN in periodicity modeling tasks (Section [4.1](https://arxiv.org/html/2410.02675v6#S4.SS1 "4.1 Periodicity Modeling ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks")). Second, we demonstrate the effectiveness and generalizability of FAN across a range of real-world tasks (Section [4.2](https://arxiv.org/html/2410.02675v6#S4.SS2 "4.2 Application of Real-world Task ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks")). Finally, we conduct further analysis of FAN (Section [4.3](https://arxiv.org/html/2410.02675v6#S4.SS3 "4.3 Further Analysis of FAN ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks")), including comparisons with Fourier-based networks, running time, hyperparameter impact, and more. See Appendix [B](https://arxiv.org/html/2410.02675v6#A2 "Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks") for more experiments and the experimental details can be found in Appendix [C](https://arxiv.org/html/2410.02675v6#A3 "Appendix C Experimental Details ‣ FAN: Fourier Analysis Networks").

![Image 5: Refer to caption](https://arxiv.org/html/2410.02675v6/x5.png)

Figure 4: Comparison of training and test losses for different models on the tasks of learning complex periodic functions.

### 4.1 Periodicity Modeling

#### Setup.

In periodic modeling tasks, we select periodic functions with practical significance and compare the models’ performance in learning the underlying principles of periodicity. Specifically, we generate data from periodic functions over a large domain, using a portion of this domain as training data and the entire domain as test data, i.e., a part of test data would be out of the domain of training data. We compare FAN and its variant FAN(Gated)§§§FAN(Gated) is a variant of FAN that adds gates to control the tendency of the layer, with the formula defined as \phi_{g}(x)=[g\cdot\cos(W_{p}x)||g\cdot\sin(W_{p}x)||(1-g)\cdot\sigma(B_{\bar{p}}+W_{\bar{p}}x)], where g is a learnable parameter., with MLP, KAN, and Transformer. The input of this task is scalar.

Results. Figure [3](https://arxiv.org/html/2410.02675v6#S3.F3 "Figure 3 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks"), as well as Figure [6](https://arxiv.org/html/2410.02675v6#A1.F6 "Figure 6 ‣ Appendix A MLP ‣ FAN: Fourier Analysis Networks") in Appendix, show the performance of FAN and other baselines in periodicity modeling. The results indicate that existing neural networks, including MLP, KAN, and Transformers, exhibit notable deficiencies in their ability to model periodicity. Although they attempt to fit these periodic functions, their ability limits their performance in modeling a large domain of periodicity, including the test data within and outside the domain of the training data. In contrast, FAN significantly outperforms baselines in all these tasks of periodicity modeling. Moreover, FAN performs exceptionally well on the test data both within and outside the domain, indicating that our specialized design of FAN can effectively model and understand periodicity rather than merely memorize the training data.

We also compare the training process of different models on the tasks of learning complex periodic functions, as shown in Figure [4](https://arxiv.org/html/2410.02675v6#S4.F4 "Figure 4 ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks"), which leads to the following findings. 1) FAN far exceeds the other baselines in both convergence speed and final effects. 2) FAN(Gated) often achieves faster convergence than FAN, but the final performance remains comparable. 3) Although the baselines show stabilization or gradual reductions in training loss as the number of epochs increases, their modeling may have diverged considerably from the distribution of the test data, resulting in a sharp increase in test loss. This phenomenon further demonstrates the shortcomings of these models in capturing periodicity.

### 4.2 Application of Real-world Task

1) Symbolic Formula Representation is a common task in both mathematics and physics. We follow the experiments conducted in KAN’s paper (Liu et al., [2024](https://arxiv.org/html/2410.02675v6#bib.bib8)), adhering to the same tasks, data, hyperparameters, and baselines. In addition to the original baselines, we also include Transformer for comparison in this task.

Results. Figure [7](https://arxiv.org/html/2410.02675v6#A1.F7 "Figure 7 ‣ Appendix A MLP ‣ FAN: Fourier Analysis Networks") in Appendix shows the performance of different models applied to common functions in mathematics and physics. We can observe that while KAN remains competitive with FAN when the number of parameters is small, its performance declines clearly as the number of parameters increases, which exhibits a U-shaped trend (Liu et al., [2024](https://arxiv.org/html/2410.02675v6#bib.bib8)). In contrast, as the number of parameters becomes large, FAN consistently outperforms the other baselines, including MLP, KAN, and Transformer, in fitting these functions, despite many of these functions being only partially periodic or even implicitly periodic. This may be attributed to FAN’s ability to capture and model both periodic and non-periodic features and the advantages of fewer parameters. These results indicate that although FAN enhances its ability to model periodicity, it does not compromise its capacity to fit non-periodic functions.

2) Time Series Forecasting plays a critical role in various real-world applications. We employ four public datasets of this task to assess the model performance on time series forecasting, including Weather (Wu et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib16)), Exchange (Lai et al., [2018](https://arxiv.org/html/2410.02675v6#bib.bib17)), Traffic (Wu et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib16)), and ETTh (Zhou et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib18)) datasets. For each dataset, we input 96 previous time steps and forecast the subsequent time steps of {96, 192, 336, 720}. In this task, we choose the sequence models as baselines, including LSTM, Mamba, and Transformer.

Table 2: Average performance on different public datasets and output lengths in time series forecasting tasks, where Input Length = 96 and the bold value indicates the best performance.

Model Num of Params Average
MSE \downarrow MAE \downarrow
LSTM 12.51M 1.083 0.726
Mamba 12.69M 1.002 0.668
Transformer 12.12M 0.994 0.689
w/ FAN(Gated)11.07M 0.845 0.637
w/ FAN 11.06M 0.839 0.631
Improvements\downarrow 1.06M\downarrow 15.6%\downarrow 8.4%

Results. As shown in Table [2](https://arxiv.org/html/2410.02675v6#S4.T2 "Table 2 ‣ 4.2 Application of Real-world Task ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks") (See Table [6](https://arxiv.org/html/2410.02675v6#A1.T6 "Table 6 ‣ Appendix A MLP ‣ FAN: Fourier Analysis Networks") in Appendix for complete results), we compare the performance of Transformer with FAN and other baselines for time series forecasting tasks. The results indicate that Transformer with FAN outperforms other representative sequence models in these tasks. The improvements of Transformer with FAN and FAN(Gated) over the standard Transformer are notable, with the average relative improvements ranging from 15.0% to 15.6% for MSE and from 7.6% to 8.4% for MAE. It suggests that incorporating explicit periodic pattern encoding within neural networks improves time series forecasting performance in real-world applications.

3) Language Modeling is a fundamental task in natural language processing. We conduct language modeling using the SST-2 (Socher et al., [2013](https://arxiv.org/html/2410.02675v6#bib.bib19)) dataset and evaluate the model’s performance on its test set, as well as on the related datasets such as IMDB (Maas et al., [2011](https://arxiv.org/html/2410.02675v6#bib.bib20)), Sentiment140 (Sahni et al., [2017](https://arxiv.org/html/2410.02675v6#bib.bib21)), and Amazon Reviews (Linden et al., [2003](https://arxiv.org/html/2410.02675v6#bib.bib22)). These four classic datasets all belong to the field of sentiment analysis. The comparisons are between Transformer with FAN and FAN(Gated), along with the classic sequence models, including LSTM, Mamba, and Transformer.

Table 3: Performance of different sequence models on language modeling tasks, where the models are trained on the training set of SST-2 and evaluated on the other datasets, the bold value indicates the best performance on each column, the bold italic indicates the second-best performance, and the improvements represent relative improvements of using FAN based on standard Transformer.

Model Num of Params SST-2 (test)IMDB Sentiment140 Amazon Reviews
Loss \downarrow Acc \uparrow Loss \downarrow Acc \uparrow Loss \downarrow Acc \uparrow Loss \downarrow Acc \uparrow
LSTM 120.14M 0.4760 80.60 0.6449 64.38 0.8026 59.79 0.5791 71.52
Mamba 129.73M 0.4335 79.59 0.6863 62.03 0.7871 58.74 0.6163 67.19
Transformer 109.48M 0.4297 81.19 0.5649 69.94 0.8891 57.79 0.5563 71.55
w/ FAN(Gated)95.33M 0.4250 80.39 0.5817 70.12 0.7941 61.94 0.4835 76.89
w/ FAN 95.32M 0.4094 81.54 0.5225 73.98 0.8257 60.93 0.4748 77.63
Improvements\downarrow 14.16M\downarrow 4.72%\uparrow 0.43%\downarrow 7.51%\uparrow 5.78%\downarrow 7.13%\uparrow 5.43%\downarrow 14.65%\uparrow 8.50%

Results. We report the performance comparison between different sequence models across four sentiment analysis datasets, as shown in Table [3](https://arxiv.org/html/2410.02675v6#S4.T3 "Table 3 ‣ 4.2 Application of Real-world Task ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks"). The results indicate that Transformer with FAN achieves clear improvements compared to the standard Transformer and other baselines, such as LSTM and Mamba, especially for zero-shot OOD performance on IMDB, Sentiment140, and Amazon Reviewers datasets. Using FAN achieves the relative improvements up to 14.65% and 8.50% in terms of Loss and Accuracy respectively, while reducing parameter numbers by about 14.16M. It indicates the potential of periodicity modeling to enhance both effectiveness and generalization on cross-domain language modeling and sentiment analysis tasks.

### 4.3 Further Analysis of FAN

Comparison with Fourier-based Networks.We compare FAN with Fourier-based networks in terms of their periodicity modeling abilities and general-purpose capabilities for language modeling. Some previous works have explored the application of Fourier-based Networks in specific tasks (Oreshkin et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib23); Tancik et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib24); Sitzmann et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib25); Han et al., [2022](https://arxiv.org/html/2410.02675v6#bib.bib26)), but these studies primarily involved shallow/small-scale models (i.e., fewer than 1M parameters). Assessing their general modeling capabilities requires evaluating their effectiveness in deeper/larger architectures, we categorize these Fourier-based networks into three main types and systematically evaluate them within the 12-layer Transformer. Specifically, we compare with: 1) Fourier Neural Network (FNN)(Silvescu, [1999](https://arxiv.org/html/2410.02675v6#bib.bib9)) using the cosine or sine function or their linear combinations as the activation function, such as SIREN (Sitzmann et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib25)). 2) Fourier Series Neural Network (FSNN) is defined as Eq. ([3](https://arxiv.org/html/2410.02675v6#S3.E3 "Equation 3 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks")), which shares the parameters and computation of sine and cosine part. 3) Fourier Transform Neural Network (FTNN) is a type of neural network that employs Fourier Transform to process the intermediate output in the neural network, such as FNO (Li et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib27)).

Figure 5: Comparison FAN with Fourier-based Networks on complex periodicity modeling (y=e^{\sin(\pi x)^{2}+\cos(x)+(x\mod 3)-1}) and language modeling.

![Image 6: Refer to caption](https://arxiv.org/html/2410.02675v6/x6.png)

Table 4: Comparison FAN with Fourier-based Networks on language modeling tasks, where each of them replaces the MLP layer in the standard transformer and ID means in-domain.

Model Num of Params Loss \downarrow
Train ID Test OOD Test
MLP 109.48M 0.2574 0.4297 0.5649
FNN 109.48M 0.6933 0.7103 0.7135
FSNN 95.32M 0.6931 0.7210 0.7249
FTNN 300.56M 0.2449 0.4547 0.8128
FAN 95.32M 0.2434 0.4094 0.5225

As shown in Figure [4.3](https://arxiv.org/html/2410.02675v6#S4.SS3 "4.3 Further Analysis of FAN ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks"), only FAN achieves excellent performance on both tasks, indicating the superiority of our specially designed architecture of FAN. In contrast, FNN and FSNN cannot fit language modeling tasks, which aligns with previous work (Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib12); Liu et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib11)) and our findings derived from Eq. ([3](https://arxiv.org/html/2410.02675v6#S3.E3 "Equation 3 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks"))-([5](https://arxiv.org/html/2410.02675v6#S3.E5 "Equation 5 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks")). Moreover, FTNN performs poorly on complex periodic modeling tasks, akin to MLP. This may be attributed to the fact that FTNN does not incorporate the Fourier principle into the network but applies Fourier Transform as an intermediate processing step, which disadvantages FTNN in capturing periodicity. From Table [4](https://arxiv.org/html/2410.02675v6#S4.T4 "Table 4 ‣ 4.3 Further Analysis of FAN ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks"), FAN also achieves fewer parameters and better performance than FTNN in language modeling tasks.

Table 5: Comparison of actual runtime between FAN and MLP under different input/output dimensions (e.g., 8192×8192 indicates both input and output dimensions are 8192).

1024\times 1024 2048\times 2048 4096\times 4096 8192\times 8192
MLP 0.064 ms 0.114 ms 0.212 ms 0.938 ms
FAN 0.128 ms 0.133 ms 0.211 ms 0.704 ms

#### Runtime of FAN.

We analyze the actual running time of FAN layer compared to MLP Layer with different input and output dimensions, as shown in Table [5](https://arxiv.org/html/2410.02675v6#S4.T5 "Table 5 ‣ 4.3 Further Analysis of FAN ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks"). The experimental results show that MLPs exhibit smaller runtimes when the input and output sizes are small, due to PyTorch’s optimization of MLP. However, as the input and output sizes continue to increase, matrix computations become the main contributor to runtime. At this point, FAN’s fewer parameters and reduced FLOPs begin to show significant advantages. Note that FAN can be further optimized from the underlying implementation.

#### The impact of hyperparameter \mathbf{d_{\text{p}}}.

In our experiments, we fix d_{\text{p}}=\frac{1}{4}d_{h} intuitively for FAN, where d_{h} denotes the dimension of hidden layers. As shown in Figure [8](https://arxiv.org/html/2410.02675v6#A2.F8 "Figure 8 ‣ B.7 The influence of hyperparameters 𝐝_\"p\" ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks") of Appendix, we investigate the impact of varying d_{\text{p}} empirically on task performance by changing itself. The results indicate that performance initially improves as d_{\text{p}} increases, but then decreases beyond a certain point. This trend may be attributed to the number of potential periodic features specific to each task. Furthermore, there remains room for further improvements with the better setup of \mathbf{d_{\text{p}}}.

## 5 Related Work

In this section, we outline the two most relevant directions and associated papers of this work.

#### Learning Periodicity with Neural Networks.

Periodic functions are one of the most basic functions of importance to human society and natural science(Newton, [1687](https://arxiv.org/html/2410.02675v6#bib.bib28); Osborn and Sensier, [2002](https://arxiv.org/html/2410.02675v6#bib.bib29); Kwasnicki, [2008](https://arxiv.org/html/2410.02675v6#bib.bib30); De Groot and Franses, [2012](https://arxiv.org/html/2410.02675v6#bib.bib31); Zhang et al., [2017](https://arxiv.org/html/2410.02675v6#bib.bib32)). However, commonly used neural networks, such as MLPs and transformers, struggle with modeling periodicity. This limitation is attributed to the lack of inherent “periodicity” in their inductive biases. Some previous works(Silvescu, [1999](https://arxiv.org/html/2410.02675v6#bib.bib9); Liu, [2013](https://arxiv.org/html/2410.02675v6#bib.bib10); Parascandolo et al., [2016](https://arxiv.org/html/2410.02675v6#bib.bib33); Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib12)) proposed merely using standard periodic functions themselves or their linear combinations as activation functions, which only work well on some shallow and simple models. On this basis, work (Liu et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib11)) introduced the Snake function, i.e., x+\sin^{2}(x), as the activation function. However, it can fit periodic functions to a certain extent, but its effect is limited especially for OOD scenarios, as demonstrated in Appendix [D](https://arxiv.org/html/2410.02675v6#A4 "Appendix D Comparison of FAN and Snake Activation Function ‣ FAN: Fourier Analysis Networks"). Therefore, although some previous studies have attempted to integrate periodic information into neural networks, their actual performance and range of applications remain heavily constrained.

#### Fourier-based Neural Network.

Previous studies have explored Fourier-based networks, but these networks generally perform well on specific tasks, while their performance on more general tasks tends to be poorer (Zuo and Cai, [2005](https://arxiv.org/html/2410.02675v6#bib.bib34); Tan, [2006](https://arxiv.org/html/2410.02675v6#bib.bib35); Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib12); Jiang et al., [2022](https://arxiv.org/html/2410.02675v6#bib.bib36); Chen et al., [2022](https://arxiv.org/html/2410.02675v6#bib.bib37)). Fourier Neural Networks employ the cosine (Silvescu, [1999](https://arxiv.org/html/2410.02675v6#bib.bib9); Ngom and Marin, [2021](https://arxiv.org/html/2410.02675v6#bib.bib38)) or sine function (Parascandolo et al., [2016](https://arxiv.org/html/2410.02675v6#bib.bib33); Sitzmann et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib25)) or their combination (Liu, [2013](https://arxiv.org/html/2410.02675v6#bib.bib10)) as the activation function. Some work employs Fourier Transform to process the intermediate output of network (Li et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib27); Lee-Thorp et al., [2022](https://arxiv.org/html/2410.02675v6#bib.bib39)), but they did not address the challenges of periodicity modeling. Some researches focus on leveraging the network to simulate the formula of Fourier Series (Rafajłowicz and Pawlak, [1997](https://arxiv.org/html/2410.02675v6#bib.bib40); Halawa, [2008](https://arxiv.org/html/2410.02675v6#bib.bib41); Lee et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib42)), which generally possesses a similar principle as Eq. ([3](https://arxiv.org/html/2410.02675v6#S3.E3 "Equation 3 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks")). However, this leads to the same problem as in Eq. ([5](https://arxiv.org/html/2410.02675v6#S3.E5 "Equation 5 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks")), i.e., they are hard to serve as building blocks for deep neural networks, which limits these approaches’ capabilities. More detailed discussion can be found in Appendix [G](https://arxiv.org/html/2410.02675v6#A7 "Appendix G More Detailed Discussion with Fourier-based Neural Network ‣ FAN: Fourier Analysis Networks").

In this paper, we design FAN to address these challenges, which performs exceptionally well on periodicity modeling and maintains broad applicability on real-world tasks.

## 6 Discussion

In this section, we have a broad discussion on expressive power, extrapolation capability, and application scope of FAN as follows: ❶ FAN theoretically possesses the equal expressive power as MLP since it also adheres to Universal Approximation Theorem, which guarantees its capacity for functional approximation (refer to Appendix [E](https://arxiv.org/html/2410.02675v6#A5 "Appendix E Compliance with the Universal Approximation Theorem ‣ FAN: Fourier Analysis Networks") for the detailed explanation). Moreover, FAN introduces an important enhancement by incorporating periodicity, a feature absent in MLPs. By leveraging this special design, FAN not only retains the capabilities of MLP but also enhances its ability to capture periodic characteristics in data. ❷ We observe that existing networks often exhibit divergent predictions in OOD scenarios, as shown in Figures [3](https://arxiv.org/html/2410.02675v6#S3.F3 "Figure 3 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks"), [4](https://arxiv.org/html/2410.02675v6#S4.F4 "Figure 4 ‣ 4 Experiments ‣ FAN: Fourier Analysis Networks"), and [6](https://arxiv.org/html/2410.02675v6#A1.F6 "Figure 6 ‣ Appendix A MLP ‣ FAN: Fourier Analysis Networks") for periodicity modeling tasks. In contrast, FAN demonstrates strong OOD extrapolation ability in both periodicity modeling and some real-world tasks. This extrapolation ability indicates that the network is no longer restricted to the paradigms present in training dataset, but instead exhibits a kind of “transboundary thinking”. This could be an important avenue for improving generalization and learning efficiency. ❸ Beyond tasks that explicitly require periodicity modeling, FAN also has utility in a broader range of applications, which has been evidenced by our extensive experiments on real-world tasks, such as symbolic formula representation, time series forecasting, language modeling, and image recognition, where FAN achieve competitive or superior performance than Transformers and other baselines. In fact, many machine learning tasks may harbor hidden forms of periodicity, even without explicitly including periodicity, such as mathematical operations and logic reasoning. If the neural network lacks the ability to model periodicity, it could impair the learning efficiency (Dong et al., [2025b](https://arxiv.org/html/2410.02675v6#bib.bib45)). From a deeper perspective, periodicity is not just a data feature but reflects a form of structural knowledge — one that allows for the transfer and reuse of abstract rules and principles across different contexts.

## 7 Conclusion and Future Work

In this paper, we have proposed Fourier Analysis Network (FAN), a novel network that addresses periodicity modeling in existing networks while maintaining the general-purpose modeling capability. Experimental results demonstrate that FAN successfully fit both basic and complex periodic functions, whereas other general-purpose networks failed. Moreover, using FAN exhibit clear improvements in real-world tasks, such as symbolic formula representation, time series forecasting, and language modeling, outperforming neural networks such as MLP, KAN, LSTM, Mamba, and Transformer. These promising results, especially the stronger performance and the fewer parameters and FLOPs compared to MLP, suggest its potential to become a key component of foundational models. Some works have demonstrated the superiority of using FAN in diverse tasks, including gravitational wave analysis (Zhao et al., [2024](https://arxiv.org/html/2410.02675v6#bib.bib46)), EEG-based emotion recognition (Wang et al., [2025](https://arxiv.org/html/2410.02675v6#bib.bib47)), and large language modeling (Dong et al., [2025b](https://arxiv.org/html/2410.02675v6#bib.bib45)), etc. In future work, we aim to further broaden the applicability of FAN.

## 8 Acknowledgement

This research is supported by the National Key R\&D Program under Grant No. 2023YFB4503801, the National Natural Science Foundation of China under Grant No. 62192733, 62192730, 62192731, the Major Program (JD) of Hubei Province (No.2023BAA024). Moreover, we would like to thank Lecheng Wang and Xuanming Zhang for their participation in discussions related to this work.

## References

*   Rosenblatt [1958] Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain. _Psychological review_, 65(6):386, 1958. 
*   Haykin [1998] Simon Haykin. _Neural networks: a comprehensive foundation_. Prentice Hall PTR, 1998. 
*   Hornik et al. [1989] Kurt Hornik, Maxwell B. Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. _Neural Networks_, 2(5):359–366, 1989. 
*   Vaswani et al. [2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NIPS_, pages 5998–6008, 2017. 
*   Touvron et al. [2023] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. _CoRR_, abs/2307.09288, 2023. 
*   OpenAI [2023] OpenAI. GPT-4 technical report. _CoRR_, abs/2303.08774, 2023. 
*   Gu and Dao [2023] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. _CoRR_, abs/2312.00752, 2023. 
*   Liu et al. [2024] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Y. Hou, and Max Tegmark. KAN: kolmogorov-arnold networks. _CoRR_, abs/2404.19756, 2024. 
*   Silvescu [1999] Adrian Silvescu. Fourier neural networks. In _IJCNN_, pages 488–491. IEEE, 1999. 
*   Liu [2013] Shuang Liu. Fourier neural network for machine learning. In _ICMLC_, pages 285–290. IEEE, 2013. 
*   Liu et al. [2020] Ziyin Liu, Tilman Hartwig, and Masahito Ueda. Neural networks fail to learn periodic functions and how to fix it. In _NeurIPS_, 2020. 
*   Uteuliyeva et al. [2020] Malika Uteuliyeva, Abylay Zhumekenov, Rustem Takhanov, Zhenisbek Assylbekov, Alejandro J. Castro, and Olzhas Kabdolov. Fourier neural networks: A comparative study. _Intell. Data Anal._, 24(5):1107–1120, 2020. 
*   Stein and Weiss [1971] Elias M Stein and Guido Weiss. _Introduction to Fourier analysis on Euclidean spaces_, volume 1. Princeton university press, 1971. 
*   Duoandikoetxea [2024] Javier Duoandikoetxea. _Fourier analysis_, volume 29. American Mathematical Society, 2024. 
*   Tolstov [2012] Georgi P Tolstov. _Fourier series_. Courier Corporation, 2012. 
*   Wu et al. [2021] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. _Advances in neural information processing systems_, 34:22419–22430, 2021. 
*   Lai et al. [2018] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-term temporal patterns with deep neural networks. In _SIGIR_, pages 95–104. ACM, 2018. 
*   Zhou et al. [2021] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In _AAAI_, pages 11106–11115. AAAI Press, 2021. 
*   Socher et al. [2013] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In _EMNLP_, pages 1631–1642. ACL, 2013. 
*   Maas et al. [2011] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In _ACL_, pages 142–150. The Association for Computer Linguistics, 2011. 
*   Sahni et al. [2017] Tapan Sahni, Chinmay Chandak, Naveen Reddy Chedeti, and Manish Singh. Efficient twitter sentiment classification using subjective distant supervision. In _COMSNETS_, pages 548–553. IEEE, 2017. 
*   Linden et al. [2003] Greg Linden, Brent Smith, and Jeremy York. Amazon.com recommendations: Item-to-item collaborative filtering. _IEEE Internet Comput._, 7(1):76–80, 2003. 
*   Oreshkin et al. [2020] Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: neural basis expansion analysis for interpretable time series forecasting. In _ICLR_. OpenReview.net, 2020. 
*   Tancik et al. [2020] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In _NeurIPS_, 2020. 
*   Sitzmann et al. [2020] Vincent Sitzmann, Julien N.P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In _NeurIPS_, 2020. 
*   Han et al. [2022] Bing Han, Cheng Wang, and Kaushik Roy. Oscillatory fourier neural network: A compact and efficient architecture for sequential processing. In _AAAI_, pages 6838–6846. AAAI Press, 2022. 
*   Li et al. [2021] Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In _ICLR_. OpenReview.net, 2021. 
*   Newton [1687] Isaac Newton. _Philosophiae naturalis principia mathematica_. William Dawson & Sons Ltd., London, 1687. 
*   Osborn and Sensier [2002] Denise R. Osborn and Marianne Sensier. The prediction of business cycle phases: Financial variables and international linkages. _National Institute Economic Review_, 182(1):96–105, 2002. doi: 10.1177/002795010218200110. URL [https://doi.org/10.1177/002795010218200110](https://doi.org/10.1177/002795010218200110). 
*   Kwasnicki [2008] Witold Kwasnicki. Kitchin, juglar and kuznetz business cycles revisited. _Wroclaw: Institute of Economic Sciences_, 2008. 
*   De Groot and Franses [2012] Bert De Groot and Philip Hans Franses. Common socio-economic cycle periods. _Technological Forecasting and Social Change_, 79(1):59–68, 2012. 
*   Zhang et al. [2017] Liheng Zhang, Charu Aggarwal, and Guo-Jun Qi. Stock price prediction via discovering multi-frequency trading patterns. In _Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, KDD ’17, page 2141–2149, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450348874. doi: 10.1145/3097983.3098117. URL [https://doi.org/10.1145/3097983.3098117](https://doi.org/10.1145/3097983.3098117). 
*   Parascandolo et al. [2016] Giambattista Parascandolo, Heikki Huttunen, and Tuomas Virtanen. Taming the waves: sine as activation function in deep neural networks. 2016. 
*   Zuo and Cai [2005] Wei Zuo and Lilong Cai. Tracking control of nonlinear systems using fourier neural network. In _Proceedings, 2005 IEEE/ASME International Conference on Advanced Intelligent Mechatronics._, pages 670–675. IEEE, 2005. 
*   Tan [2006] HS Tan. Fourier neural networks and generalized single hidden layer networks in aircraft engine fault diagnostics. 2006. 
*   Jiang et al. [2022] Song Jiang, Tahin Syed, Xuan Zhu, Joshua Levy, Boris Aronchik, and Yizhou Sun. Bridging self-attention and time series decomposition for periodic forecasting. In _CIKM_, pages 3202–3211. ACM, 2022. 
*   Chen et al. [2022] Hanlong Chen, Luzhe Huang, Tairan Liu, and Aydogan Ozcan. Fourier imager network (FIN): A deep neural network for hologram reconstruction with superior external generalization. _Light: Science & Applications_, 2022. 
*   Ngom and Marin [2021] Marieme Ngom and Oana Marin. Fourier neural networks as function approximators and differential equation solvers. _Stat. Anal. Data Min._, 14(6):647–661, 2021. 
*   Lee-Thorp et al. [2022] James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontañón. Fnet: Mixing tokens with fourier transforms. In _NAACL-HLT_, pages 4296–4313. Association for Computational Linguistics, 2022. 
*   Rafajłowicz and Pawlak [1997] E Rafajłowicz and M Pawlak. On function recovery by neural networks based on orthogonal expansions. _Nonlinear Analysis: Theory, Methods & Applications_, 30(3):1343–1354, 1997. 
*   Halawa [2008] Krzysztof Halawa. Fast and robust way of learning the fourier series neural networks on the basis of multidimensional discrete fourier transform. In _ICAISC_, volume 5097 of _Lecture Notes in Computer Science_, pages 62–70. Springer, 2008. 
*   Lee et al. [2021] Jiyoung Lee, Wonjae Kim, Daehoon Gwak, and Edward Choi. Conditional generation of periodic signals with fourier-based decoder. _CoRR_, abs/2110.12365, 2021. 
*   Dong et al. [2024] Yihong Dong, Xue Jiang, Huanyu Liu, Zhi Jin, Bin Gu, Mengfei Yang, and Ge Li. Generalization or memorization: Data contamination and trustworthy evaluation for large language models. In _ACL (Findings)_, pages 12039–12050. Association for Computational Linguistics, 2024. 
*   Dong et al. [2025a] Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, Zhi Jin, Fei Huang, Yongbin Li, and Ge Li. RL-PLUS: countering capability boundary collapse of llms in reinforcement learning with hybrid-policy optimization. _CoRR_, abs/2508.00222, 2025a. 
*   Dong et al. [2025b] Yihong Dong, Ge Li, Xue Jiang, Yongding Tao, Kechi Zhang, Hao Zhu, Huanyu Liu, Jiazheng Ding, Jia Li, Jinliang Deng, and Hong Mei. Fanformer: Improving large language models through effective periodicity modeling. _CoRR_, abs/2502.21309, 2025b. 
*   Zhao et al. [2024] Tianyu Zhao, Yue Zhou, Ruijun Shi, Peng Xu, Zhoujian Cao, and Zhixiang Ren. Compact binary coalescence gravitational wave signals counting and separation using unmixformer, 2024. URL [https://arxiv.org/abs/2412.18259](https://arxiv.org/abs/2412.18259). 
*   Wang et al. [2025] Jinfeng Wang, Yanhao Huang, Sifan Song, Boqian Wang, Jionglong Su, and Jiaman Ding. A novel fourier adjacency transformer for advanced eeg emotion recognition, 2025. URL [https://arxiv.org/abs/2503.13465](https://arxiv.org/abs/2503.13465). 
*   LeCun et al. [2010] Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. _ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist_, 2, 2010. 
*   Ganin et al. [2016] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor S. Lempitsky. Domain-adversarial training of neural networks. _J. Mach. Learn. Res._, 17:59:1–59:35, 2016. 
*   Xiao et al. [2017] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. _CoRR_, abs/1708.07747, 2017. URL [http://arxiv.org/abs/1708.07747](http://arxiv.org/abs/1708.07747). 
*   Weiss and Tonella [2022] Michael Weiss and Paolo Tonella. Simple techniques work surprisingly well for neural network test prioritization and active learning. In _Proceedings of the 31th ACM SIGSOFT International Symposium on Software Testing and Analysis_, 2022. 
*   Zhou et al. [2022] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In _ICML_, volume 162 of _Proceedings of Machine Learning Research_, pages 27268–27286. PMLR, 2022. 
*   Ulyanov et al. [2016] Dmitry Ulyanov, Andrea Vedaldi, and Victor S. Lempitsky. Instance normalization: The missing ingredient for fast stylization. _CoRR_, abs/1607.08022, 2016. 
*   Hochreiter and Schmidhuber [1997] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. _Neural Comput._, 9(8):1735–1780, 1997. 
*   LeCun et al. [1998] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. _Proc. IEEE_, 86(11):2278–2324, 1998. 
*   Hendrycks and Gimpel [2016] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). _arXiv preprint arXiv:1606.08415_, 2016. 
*   Loshchilov and Hutter [2019] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR (Poster)_. OpenReview.net, 2019. 
*   Devlin et al. [2018] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. _CoRR_, abs/1810.04805, 2018. URL [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805). 
*   Wan et al. [2013] Li Wan, Matthew D. Zeiler, Sixin Zhang, Yann LeCun, and Rob Fergus. Regularization of neural networks using dropconnect. In _ICML (3)_, volume 28 of _JMLR Workshop and Conference Proceedings_, pages 1058–1066. JMLR.org, 2013. 
*   Belcak and Wattenhofer [2022] Peter Belcak and Roger Wattenhofer. Periodic extrapolative generalisation in neural networks. In _SSCI_, pages 1066–1073. IEEE, 2022. 

## Appendix A MLP

The MLP layer \Phi(x) is defined as:

\Phi(x)=\sigma(B_{m}+W_{m}x),(12)

where B_{m}\in\mathbb{R}^{d_{m}} and W_{\bar{p}}\in\mathbb{R}^{d_{x}\times d_{m}} are learnable parameters with the hyperparameter d_{m} indicating the first dimension of W_{m}, \sigma denotes the activation function, and MLP can be defined as the stacking of the MLP layer \Phi(x):

\text{MLP}(x)=\Phi_{L}\circ\Phi_{L-1}\circ\cdots\circ\Phi_{1}\circ x,(13)

where

\Phi_{l}(x)=\left\{\begin{array}[]{ll}\sigma(B_{m}^{l}+W^{l}_{m}x),&\text{if }l<L,\\
B^{L}+W^{L}x,&\text{if }l=L.\end{array}\right.(14)

![Image 7: Refer to caption](https://arxiv.org/html/2410.02675v6/x7.png)

![Image 8: Refer to caption](https://arxiv.org/html/2410.02675v6/x8.png)

![Image 9: Refer to caption](https://arxiv.org/html/2410.02675v6/x9.png)

Figure 6: The performance of FAN in periodicity modeling compared to MLP, KAN, and Transformer (Part II), where the green line represents the test data within the domain of training data, while the blue line represents the test data outside the domain of training data.

![Image 10: Refer to caption](https://arxiv.org/html/2410.02675v6/x10.png)

Figure 7: Comparisons of FAN with the baselines, including MLP, KAN, and Transformer, across varying numbers of parameters on symbolic formula representation tasks.

Table 6: Performance of different sequence models on time series forecasting tasks, where Input Length = 96, the bold values indicate the lowest value on each row, and Improve means the relative improvements of using FAN and FAN(Gated) based on standard Transformer.

Dataset Output Length LSTM (12.51 M)Mamba (12.69 M)Transformer (12.12 M)Transformer with FAN(11.06 M)
Gated Default
MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow
Weather 96 1.069 0.742 0.552 0.519 0.413 0.438 0.292 0.380 0.313 0.431
192 1.090 0.778 0.700 0.595 0.582 0.540 0.535 0.550 0.472 0.525
336 0.992 0.727 0.841 0.667 0.751 0.626 0.637 0.602 0.719 0.581
720 1.391 0.892 1.171 0.803 0.967 0.715 0.845 0.706 0.732 0.670
Exchange 96 0.938 0.794 0.908 0.748 0.777 0.681 0.685 0.644 0.657 0.623
192 1.241 0.899 1.328 0.925 1.099 0.800 0.998 0.757 0.968 0.741
336 1.645 1.048 1.512 0.992 1.614 1.029 1.511 0.961 1.266 0.905
720 1.949 1.170 2.350 1.271 2.163 1.204 1.658 1.104 1.857 1.145
Traffic 96 0.659 0.359 0.666 0.377 0.656 0.357 0.647 0.355 0.643 0.347
192 0.668 0.360 0.671 0.381 0.672 0.363 0.649 0.353 0.657 0.354
336 0.644 0.342 0.665 0.374 0.673 0.360 0.665 0.358 0.656 0.353
720 0.654 0.351 0.662 0.364 0.701 0.380 0.682 0.369 0.673 0.363
ETTh 96 0.999 0.738 0.860 0.697 1.139 0.853 0.842 0.736 0.873 0.707
192 1.059 0.759 0.849 0.700 1.373 0.932 0.885 0.748 0.914 0.741
336 1.147 0.820 1.005 0.745 1.261 0.924 0.980 0.770 0.999 0.793
720 1.206 0.847 0.994 0.758 1.056 0.819 1.002 0.798 1.031 0.818
Average (Improve)–1.083 0.726 1.002 0.668 0.994 0.689 0.845 \downarrow 15.0%0.637 \downarrow 7.6%0.839 \downarrow 15.6%0.631 \downarrow 8.4%

## Appendix B Additional Experiments

### B.1 Additional Experiments on Periodicity Modeling Tasks.

More experimental results on periodicity modeling tasks are shown in Figure [6](https://arxiv.org/html/2410.02675v6#A1.F6 "Figure 6 ‣ Appendix A MLP ‣ FAN: Fourier Analysis Networks").

### B.2 Additional Experiments on Image Recognition Tasks.

Image Recognition is a key computer vision task where image content is identified and categorized. Our evaluation contains four public benchmarks of image recognition: MNIST [LeCun et al., [2010](https://arxiv.org/html/2410.02675v6#bib.bib48)], MNIST-M[Ganin et al., [2016](https://arxiv.org/html/2410.02675v6#bib.bib49)], Fashion-MNIST[Xiao et al., [2017](https://arxiv.org/html/2410.02675v6#bib.bib50)], and Fashion-MNIST-C[Weiss and Tonella, [2022](https://arxiv.org/html/2410.02675v6#bib.bib51)], where MNIST-M and Fashion-MNIST-C are the variants for robustness.

Table 7: Results on image recognition tasks, where OOD Accuracy means the performance on other paired datasets and the Bold values indicate the highest values under the same metric.

Dataset Accuracy \uparrow OOD Accuracy \uparrow
CNN w/ FAN CNN w/ FAN
MNIST 99.63 99.67 28.85 30.3
MNIST-M 94.52 94.23 82.85 83.55
Fashion-MNIST 94.15 94.47 49.82 51.88
Fashion-MNIST-C 88.61 88.82 91.45 91.59

Results. We apply FAN to image recognition tasks on four classic benchmarks, as shown in Table [7](https://arxiv.org/html/2410.02675v6#A2.T7 "Table 7 ‣ B.2 Additional Experiments on Image Recognition Tasks. ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks"). The results show that using FAN outperforms the standard CNN in most cases. We believe that there are also some latent periodic features in image recognition tasks, and FAN’s ability to model these periodic features can help CNN achieve competitive or superior performance, especially in OOD scenarios.

### B.3 Evaluation on LLMs with FAN

Table [8](https://arxiv.org/html/2410.02675v6#A2.T8 "Table 8 ‣ B.3 Evaluation on LLMs with FAN ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks") reports the zero-shot results on the LM Eval Harness benchmark. The results show that using FAN outperforms standard Transformer architecture across various tasks with the same training tokens of 200B.

Table 8: Comparison of our approach with well-trained Transformer language models on LM Eval Harness benchmark. Both of them are trained on 200B tokens and using FAN achieves better accuracy.

Models arc challenge arc easy boolq hella-swag open bookqa piqa sciq wino-grande avg.
Transformer-1B 29.7 63.3 59.6 52.5 34.6 71.4 85.8 55.9 56.6
Ours 32.1 63.5 60.1 53.8 34.7 72.5 89.9 56.1 57.9

### B.4 FAN for Solving SciML Problems

We conduct experiments on the SciML problem that includes the Fourier function class following the work [Li et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib27)]. The Burgers’ equation, a non-linear partial differential equation, is frequently used in scientific computing to model shock waves and traffic flow, among other phenomena. The detailed error rate on Burgers’ equation is listed in the Table [9](https://arxiv.org/html/2410.02675v6#A2.T9 "Table 9 ‣ B.4 FAN for Solving SciML Problems ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks"). We can find that replacing the MLP Layer with FAN Layer in Fourier Neural Operator (FNO) [Li et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib27)] can achieve clear improvements on each setting of resolution s of this task.

Table 9: The error rate on Burgers’ equation. The values in the table represent the Average Relative Error for Burgers’ equation with lower values indicating better performance.

Model s=256 s=512 s=1024 s=2048 s=4096 s=8192
FNO 5.93%6.14%6.03%6.75%7.36%9.93%
FNO with FAN 5.26%5.17%5.18%6.73%6.35%7.06%

### B.5 Comparison with Frequency-based Models in Time Series Forecasting Tasks

To compare with frequency-based models in Time Series Forecasting tasks such as FEDformer [Zhou et al., [2022](https://arxiv.org/html/2410.02675v6#bib.bib52)], we replace MLP with FAN in frequency-based models. We present the experimental results in Table [10](https://arxiv.org/html/2410.02675v6#A2.T10 "Table 10 ‣ B.5 Comparison with Frequency-based Models in Time Series Forecasting Tasks ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks"), where the results of FEDformer are cited from its paper directly. From the results, we can find that FEDformer with FAN can outperform FEDformer in almost all cases.

Table 10: Results of comparison with frequency-based models in time series forecasting tasks.

Dataset Len FEDformer with FAN
MSE MAE MSE MAE
Traffic 96 0.587 0.366 0.577 0.357
192 0.604 0.373 0.601 0.366
336 0.621 0.383 0.620 0.378
720 0.626 0.382 0.619 0.370
Exchange 96 0.148 0.278 0.138 0.267
192 0.271 0.380 0.261 0.371
336 0.460 0.500 0.461 0.503
720 1.195 0.841 1.159 0.827
Electricity 96 0.193 0.308 0.184 0.298
192 0.201 0.315 0.199 0.313
336 0.214 0.329 0.212 0.325
720 0.246 0.355 0.239 0.347

### B.6 Comparison with Directly Learning the Coefficients

We compare FAN with a baseline of directly learning the coefficients, which inputs sin(x) and cos(x) and then uses the MLP Layer instead of the FAN Layer to model the Fourier coefficients. In this setting, frequencies are fixed and only the coefficients are learned, which may limit the model’s ability to capture patterns not aligned with these frequencies. Taking simple f(x)=x\ mod\ 5 as an example, this setting may not even converge at all, because the frequency of x\ mod\ 5 is inconsistent with sin(x) and cos(x). The experimental results of their loss are shown in Table [11](https://arxiv.org/html/2410.02675v6#A2.T11 "Table 11 ‣ B.6 Comparison with Directly Learning the Coefficients ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks").

Table 11: Comparison of FAN and directly learning the coefficients on fitting f(x)=x\ mod\ 5.

Epoch 50 100 150 200
Directly learning the coefficients 2.10 2.09 2.09 2.08
FAN 0.28 0.23 0.18 0.17

### B.7 The influence of hyperparameters \mathbf{d_{\text{p}}}

We evaluate the influence of hyperparameters \mathbf{d_{\text{p}}} as shown in Figure [8](https://arxiv.org/html/2410.02675v6#A2.F8 "Figure 8 ‣ B.7 The influence of hyperparameters 𝐝_\"p\" ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks").

![Image 11: Refer to caption](https://arxiv.org/html/2410.02675v6/x11.png)

Figure 8: The influence of hyper-parameters \mathbf{d_{\text{p}}} on language modeling tasks. We use the red dashed line to represent the performance of the standard Transformer.

### B.8 The effectiveness of the FAN Layer for deep neural networks

We evaluate the effect of varying the number of FAN layers from 3 to 20 on periodicity modeling tasks, employing residual connections to mitigate overfitting. The experimental results show that both the best training loss and test loss still decrease slowly as the number of layers increases.

Furthermore, on Language Modeling tasks, we replaced 24 MLP Layers of Transformer with 24 FAN Layers, i.e. Transformer with FAN, and it also achieved clear improvements on each task, especially for OOD zero-shot evaluation scenarios. These findings indicate that FAN Layer is effective for deep neural networks.

![Image 12: Refer to caption](https://arxiv.org/html/2410.02675v6/x12.png)

Figure 9: Performance of Deeper FAN on fitting y=e^{\sin^{2}(\pi x)+\cos(x)+(x\mod 3)}-1.

### B.9 Experiments on Time Series Forecasting with Instance Normalization

We conduct experiments on time series forecasting tasks with instance normalization [Ulyanov et al., [2016](https://arxiv.org/html/2410.02675v6#bib.bib53)], and the results are shown in Table [12](https://arxiv.org/html/2410.02675v6#A2.T12 "Table 12 ‣ B.9 Experiments on Time Series Forecasting with Instance Normalization ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks"). We find that applying instance normalization before the architecture can effectively improve the performance.

Table 12: Results on time series forecasting tasks with instance normalization, where Input Length = 96, the bold values indicate the lowest value on each row, and the improve means the relative improvements of using FAN and FAN(Gated) based on Transformer.

Dataset Output Length Transformer (12.12 M)Transformer with FAN(11.06 M)
Gated Default
MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow
Weather 96 0.1772 0.2301 0.1864 0.2352 0.1756 0.2247
192 0.2438 0.2844 0.2445 0.2834 0.2327 0.2760
336 0.3077 0.3267 0.3156 0.3320 0.3118 0.3291
720 0.4253 0.3982 0.3909 0.3782 0.4113 0.3906
Exchange 96 0.1433 0.2653 0.1157 0.2452 0.1436 0.2666
192 0.2563 0.3552 0.2539 0.3611 0.2651 0.3757
336 0.5273 0.5218 0.4329 0.4891 0.5092 0.5326
720 1.7401 0.9273 1.5783 0.9303 1.0599 0.7657
Traffic 96 0.6160 0.3449 0.6030 0.3334 0.6109 0.3319
192 0.6329 0.3479 0.6239 0.3404 0.6258 0.3370
336 0.6369 0.3485 0.6416 0.3487 0.6200 0.3380
720 0.6555 0.3577 0.6645 0.3574 0.6412 0.3525
ETTh 96 0.3881 0.4097 0.4082 0.4292 0.3833 0.4149
192 0.5766 0.4999 0.4695 0.4514 0.5039 0.4640
336 0.5782 0.5100 0.5556 0.5012 0.5417 0.4940
720 0.5841 0.5230 0.5070 0.4943 0.5272 0.4951
Average (Improve)–0.531 0.416 0.499 \downarrow 6.1%0.406 \downarrow 2.2%0.472 \downarrow 11.0%0.399 \downarrow 4.1%

### B.10 Layer-wise Spectral Analysis

We conduct experiments on layer-wise spectral analysis below. We perform a Fast Fourier Transform (FFT) on each layer’s outputs and calculate four key metrics to quantify the spectral characteristics:

1.   1.Spectral Centroid: Measures the ”center of mass” of the spectrum, indicating whether the layer’s features are concentrated in low or high-frequency regions. 
2.   2.Spectral Sparsity (L1/L2 Norm): Quantifies how concentrated the spectral energy is within a few frequency bins. A higher value implies a more structured and less noisy signal. 
3.   3.Spectral Entropy: Measures the uniformity and predictability of the spectrum. A lower entropy indicates a more ordered and well-defined spectral structure. 
4.   4.Dominant Energy Ratio (Top-5): The proportion of total spectral energy contained within the top 5 most dominant frequency components, indicating how focused the representation is on key periodic features. 

The results reveal a highly effective multi-stage learning process, which is more sophisticated than a simple monotonic evolution of frequencies. We observe a clear three-stage “Deconstruction-Exploration-Reconstruction” mechanism:

1.   1.Initial Approximation (Layer 1): The first layer rapidly forms an initial, highly-focused approximation of the signal, as shown by its very high Dominant Energy Ratio (96.1%). 
2.   2.Feature Deconstruction and Exploration (Layers 2–8): To model the function’s complex, non-sinusoidal components (especially the x\pmod{3} term, which requires a wide range of Fourier series terms), the intermediate layers must first “deconstruct” the signal. This is evidenced by a sharp increase in Spectral Entropy and a decrease in the Dominant Energy Ratio. The network actively disperses energy across a broader spectrum to explore and capture these challenging features, showcasing the flexibility afforded by its depth. 
3.   3.Integration and Reconstruction (Layers 9–11): In the final layers, the model’s task shifts from exploration to integration. It “reconstructs” a final, efficient representation from the features learned in the middle layers. This is marked by a dramatic decrease in both Spectral Entropy and Spectral Centroid, alongside a sharp increase in the Dominant Energy Ratio to a final value of 93.8%. The network converges to a “clean”, low-frequency, and highly structured representation that is optimal for the final linear layer to map to the target output. 

Table 13: Layer-wise spectral analysis of FAN layer outputs.

Layer Spectral Centroid Spectral Sparsity Spectral Entropy Dominant Energy Ratio (Top-5)
FAN Layer 1 4.1213 3.4767 1.2264 0.9612
FAN Layer 2 2.8760 5.0003 3.2549 0.7602
FAN Layer 3 2.8804 5.0626 3.1556 0.7807
FAN Layer 4 2.8810 4.7149 2.6616 0.8426
FAN Layer 5 3.0820 4.5832 2.2248 0.8753
FAN Layer 6 3.0815 5.2388 2.5560 0.8378
FAN Layer 7 2.6955 5.8367 3.0115 0.7806
FAN Layer 8 2.9132 5.5387 2.7301 0.8086
FAN Layer 9 2.7376 4.1371 1.6760 0.8986
FAN Layer 10 2.1266 3.1509 1.0673 0.9356
FAN Layer 11 1.7721 2.9775 0.9270 0.9375

### B.11 Ablation Study

We conduct ablation studies on just cosine function, having FAN layers only in part of the network, and freezing W_{p}. The results show that FAN demonstrates a clear advantage over the variants in Periodicity Modeling and Language Modeling tasks.

Table 14: Results for ablation studies on the Periodicity Modeling task.

Periodicity Modeling Epoch=0 Epoch=100 Epoch=1000
training loss test loss training loss test loss training loss test loss
FAN_cos 39.85 63.42 2.67 10.18 1.80 5.26
FAN_replace_first_1/3_part 39.54 46.15 2.95 6.81 1.37 45.44
FAN_replace_last_1/3_part 42.82 55.46 21.96 27.86 22.79 30.51
freezing W_{p} for FAN 40.52 60.20 15.57 89.09 1.13 156.25
FAN 39.62 61.02 2.75 7.43 1.05 4.15

Table 15: Results for ablation studies on the Language Modeling task.

Language Modeling Train Loss In-domain Test Loss OOD Test Loss
FAN_cos 0.2419 0.4802 0.7727
FAN_replace_first_1/3_part 0.2693 0.4313 0.6700
FAN_replace_last_1/3_part 0.2417 0.4660 0.8052
freezing W_{p} for FAN 0.2376 0.4736 0.6324
FAN 0.2434 0.4094 0.6077

## Appendix C Experimental Details

#### Baselines.

In our experiments, we mainly compare FAN with the following baselines: 1) MLP[Rosenblatt, [1958](https://arxiv.org/html/2410.02675v6#bib.bib1)], 2) Transformer[Vaswani et al., [2017](https://arxiv.org/html/2410.02675v6#bib.bib4)], 3) KAN[Liu et al., [2024](https://arxiv.org/html/2410.02675v6#bib.bib8)], 4) LSTM[Hochreiter and Schmidhuber, [1997](https://arxiv.org/html/2410.02675v6#bib.bib54)], 5) Mamba[Gu and Dao, [2023](https://arxiv.org/html/2410.02675v6#bib.bib7)], 6) CNN[LeCun et al., [1998](https://arxiv.org/html/2410.02675v6#bib.bib55)]. Details of the baselines are given in Appendix [F](https://arxiv.org/html/2410.02675v6#A6 "Appendix F More Details of Baselines ‣ FAN: Fourier Analysis Networks"). Moreover, we also include the following variants of FAN into our comparisons: I) FAN(Gated): a variant of FAN that adds gates to control the tendency of the layer, with the formula defined as \phi_{g}(x)=[g\cdot\cos(W_{p}x)||g\cdot\sin(W_{p}x)||(1-g)\cdot\sigma(B_{\bar{p}}+W_{\bar{p}}x)], where g is a learnable parameter. II) Transformer with FAN and Transformer with FAN(Gated): we replace each MLP layer in Transformer with the FAN layer computed via Eq. ([9](https://arxiv.org/html/2410.02675v6#S3.E9 "Equation 9 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks")) and the layer of FAN(Gated), respectively. III) CNN with FAN: similarly, we replace each MLP layer in CNN with the FAN layer.

### C.1 Implementation Details.

We conduct our experiments on a single GPU of Tesla A100-PCIe-40G. Unless otherwise specified, we use the following hyperparameters in the experiments. The model architecture consists of 3 to 24 layers, the activation function \sigma is set to GELU [Hendrycks and Gimpel, [2016](https://arxiv.org/html/2410.02675v6#bib.bib56)], and the dimension of the projection matrix W_{p} is set to d_{p}=\frac{1}{4}d_{h}, where d_{h} denotes the dimension of the hidden layers. We employ the AdamW optimizer [Loshchilov and Hutter, [2019](https://arxiv.org/html/2410.02675v6#bib.bib57)] for the model’s training process.

### C.2 Setup of Periodicity Modeling

In periodicity modeling tasks, FAN, MLP, and KAN each consist of three layers with comparable FLOPs, while the Transformer model comprises twelve layers. For consistency, we set the hidden layer dimension (d_{h}) to 2048 for FAN, MLP, and Transformer. In the case of KAN, we follow its original paper [Liu et al., [2024](https://arxiv.org/html/2410.02675v6#bib.bib8)], where the spline order (K) and the number of spline intervals (G) are set to 3 and 50, respectively. We apply a learning rate of 1\times 10^{-5} for training all models. We ensured that the data density of each period in tasks was consistent, meaning that each cycle contained a fixed quantity of 10,000 training data points.

### C.3 Setup of Symbolic Formula Representation

In symbolic formula representation tasks, we used the create_dataset function from the official KAN repository to generate the datasets. Each dataset contains 3000 training samples and 1000 test samples, with all input variables randomly sampled from the range [-1, 1]. We followed the training settings from the original KAN paper, training all methods using LBFGS and Adam for 1800 steps, and selecting the best-performing result from the two optimization approaches. For KAN, we increased the number of grid points to scale up the parameter size, covering G=\{3,5,10,20,50,100,200,500,1000\}. For other methods, we scaled up the parameter size by increasing the number of layers and the dimensions of hidden layers.

### C.4 Setup of Time Series Forecasting

In time series forecasting task, we implement our model based on the codebase by [Wu et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib16)]. Each model comprises 2 encoder layers and 1 decoder layer. We fix the hidden size for both the Transformer and our model at 512, with the feedforward dimension set to 2048 (four times the hidden size). The parameter sizes detailed in the main text correspond to the Exchange dataset; variations in the number of variables across different datasets influence the linear layers in the model. We adjust the hidden sizes of the other models to align with the Transformer parameters for fairness.

### C.5 Setup of Language Modeling

In language modeling task, we employ the BERT tokenizer [Devlin et al., [2018](https://arxiv.org/html/2410.02675v6#bib.bib58)] and an embedding layer with a dimensionality of 768, except for Mamba, which adheres to its default settings as specified in the original paper [Gu and Dao, [2023](https://arxiv.org/html/2410.02675v6#bib.bib7)]. The architecture features 4, 24, and 12 layers with hidden sizes of 1800, 768, and 768 for LSTM, Mamba, and Transformers, respectively. To mitigate training stagnation in deeper LSTM models, we reduce the number of layers while increasing the hidden size to balance the parameters. Importantly, Mamba’s layer count is twice that of a similarly sized Transformer, as each layer consists of two Mamba blocks (Multihead attention block + MLP block).

### C.6 Setup of Image Recognition

In image recognition tasks, we employ the CNN as the baseline model, which consists of four Convolutional Layers and two MLP Layers (It achieves a 0.37% error rate on MNIST without augmentation, outperforming the SOTA CNN’s 0.63% [Wan et al., [2013](https://arxiv.org/html/2410.02675v6#bib.bib59)]). We replace MLP with FAN in CNN, i.e., CNN with FAN, as the counterpart, ensuring that they have similar parameters. For each task, we use stochastic gradient descent with momentum (SGDM) as the optimizer, the learning rate is set to 0.01, and the training process runs for 100 epochs.

![Image 13: Refer to caption](https://arxiv.org/html/2410.02675v6/x13.png)

![Image 14: Refer to caption](https://arxiv.org/html/2410.02675v6/x14.png)

Figure 10: Comparisons of FAN with MLP (Snake) [Liu et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib11)] in fitting periodic functions.

## Appendix D Comparison of FAN and Snake Activation Function

We compare FAN with Snake, a previous approach used for improving the fitting of periodic functions with neural networks. The results are shown in Figure [10](https://arxiv.org/html/2410.02675v6#A3.F10 "Figure 10 ‣ C.6 Setup of Image Recognition ‣ Appendix C Experimental Details ‣ FAN: Fourier Analysis Networks").

## Appendix E Compliance with the Universal Approximation Theorem

The Universal Approximation Theorem asserts that a feed-forward network with a single hidden layer, containing a sufficiently large and finite number of neurons, can approximate any continuous function defined on compact subsets of \mathbb{R}^{n}, provided that the activation function is non-constant, continuous, and nonlinear. In the case of the Fourier Analysis Network (FAN) layer, we define the mapping as:

\phi(x)=\left[\cos(W_{p}x)\;\Big\|\;\sin(W_{p}x)\;\Big\|\;\sigma(B_{\bar{p}}+W_{\bar{p}}x)\right],

where || denotes concatenation, and \sigma(\cdot) represents a standard nonlinear activation function, such as ReLU or GELU. The components \cos(W\_px) and \sin(W\_px) are non-constant, continuous, and nonlinear functions, satisfying the requisite conditions for an activation function in the Universal Approximation Theorem. Therefore, the FAN layer conforms to the Universal Approximation Theorem, enabling it to approximate arbitrary continuous functions on compact subsets of \mathbb{R}^{n}.

This proof demonstrates that the FAN layer, through its periodic components (sine and cosine functions) and the nonlinear activation \sigma(\cdot), satisfies the key conditions of the Universal Approximation Theorem, ensuring its capability to approximate complex functional mappings.

## Appendix F More Details of Baselines

In our experiments, we mainly compare FAN with the following baselines. 1) MLP[Rosenblatt, [1958](https://arxiv.org/html/2410.02675v6#bib.bib1)]: the most classic model, which is widely used in the backbone of various models. 2) Transformer[Vaswani et al., [2017](https://arxiv.org/html/2410.02675v6#bib.bib4)]: a prevalent model known for its self-attention mechanism, which achieves outstanding performance on various tasks. 3) KAN[Liu et al., [2024](https://arxiv.org/html/2410.02675v6#bib.bib8)]: an emerged model specialized for symbolic formula representation, which uses the b-spline functions instead of fixed activation functions. 4) LSTM[Hochreiter and Schmidhuber, [1997](https://arxiv.org/html/2410.02675v6#bib.bib54)]: a well-known recurrent neural network (RNN) that can capture long-term dependencies on sequential data. 5) Mamba[Gu and Dao, [2023](https://arxiv.org/html/2410.02675v6#bib.bib7)]: an emerged selective state space model (SSM) that achieves competitive performance on some tasks with sequential inputs. 6) CNN[LeCun et al., [1998](https://arxiv.org/html/2410.02675v6#bib.bib55)]: convolutional neural network contains the convolutional layers, which are effective in processing image data.

For Fourier-based Networks, we mainly compare FAN with 1) Fourier Neural Network (FNN) [Silvescu, [1999](https://arxiv.org/html/2410.02675v6#bib.bib9)] using the cosine or sine function or their linear combinations as the activation function, such as SIREN [Sitzmann et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib25)]. 2) Fourier Series Neural Network (FSNN) is defined as Eq. ([3](https://arxiv.org/html/2410.02675v6#S3.E3 "Equation 3 ‣ 3 Fourier Analysis Network (FAN) ‣ FAN: Fourier Analysis Networks")), which shares the parameters and computation of Sin and Cos part. 3) Fourier Transform Neural Network (FTNN) is a type of neural network that employs Fourier Transform to process the intermediate output in the neural network, such as FNO [Li et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib27)].

## Appendix G More Detailed Discussion with Fourier-based Neural Network

For FNNs [Silvescu, [1999](https://arxiv.org/html/2410.02675v6#bib.bib9), Liu, [2013](https://arxiv.org/html/2410.02675v6#bib.bib10), Parascandolo et al., [2016](https://arxiv.org/html/2410.02675v6#bib.bib33), Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib12)], they face challenges in scaling to deeper networks, i.e., the capacity of their deep networks to fit the Fourier coefficients is independent of the network depth, as analyzed in Section 3. The depth scalability limits their applicability to more complex, general-purpose tasks such as language modeling. Our core differences are, ”we design FAN based on the following principles: 1) the capacity of FAN to represent the Fourier coefficients should be positively correlated to its depth; 2) the output of any hidden layer can be employed to model periodicity using Fourier Series through the subsequent layers.” In Section 4.3, we conduct experiments to compare our approach with FNNs, and FNNs cannot fit language modeling tasks, but our approach works well. We provide the analysis of FNNs compared to FAN below. We mainly discuss the work [Silvescu, [1999](https://arxiv.org/html/2410.02675v6#bib.bib9), Liu, [2013](https://arxiv.org/html/2410.02675v6#bib.bib10), Parascandolo et al., [2016](https://arxiv.org/html/2410.02675v6#bib.bib33)], due to the work [Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v6#bib.bib12)] is a comparative study without proposing a new method.

Table 16: Comparison of parameters and FLOPs for different layers, where d_{i}=d_{\text{input}} (Input dimension hyperparameter), d_{o}=d_{\text{output}} (Output dimension hyperparameter), d_{p} = FAN layer hyperparameter (default \frac{1}{4}d_{o}), d_{h} = Hidden dimension hyperparameter, d_{c},d_{s} = Layer hyperparameters of cosine/sine branch dimensions, m = Layer hyperparameter of projections number, \gamma = FLOPs per nonlinear activation (\sigma, \cos, or \sin).

Metric FAN Layer Layer of [Silvescu, [1999](https://arxiv.org/html/2410.02675v6#bib.bib9)]Layer of [Liu, [2013](https://arxiv.org/html/2410.02675v6#bib.bib10)]Layer of [Parascandolo et al., [2016](https://arxiv.org/html/2410.02675v6#bib.bib33)]
Formula[\cos(W_{p}x)\parallel\sin(W_{p}x)\parallel\sigma(B_{\bar{p}}+W_{\bar{p}}x)]W_{f}\prod_{m}\cos(W_{a_{m}}x+B_{a_{m}})+B_{f}W_{f_{c}}\cos(W_{a_{c}}x+B_{a_{c}})+B_{f_{c}}+W_{f_{s}}\sin(W_{a_{s}}x+B_{a_{s}})+B_{f_{s}}W_{f}\sin(W_{a}x+B_{a})+B_{f}
Num Params(1-\frac{d_{p}}{d_{o}})(d_{i}d_{o}+d_{o})m(d_{i}d_{h}+d_{h})+d_{o}d_{h}+d_{o}d_{i}(d_{c}+d_{s})+(d_{c}+d_{s})+d_{o}(d_{c}+d_{s})+2d_{o}d_{i}d_{h}+d_{h}+d_{o}d_{h}+d_{o}
FLOPs(1-\frac{d_{p}}{d_{o}})\times 2d_{i}d_{o}+\gamma d_{o}2md_{h}d_{i}+d_{h}(m-1)+2d_{h}d_{o}+\gamma md_{h}2d_{i}(d_{c}+d_{s})+2d_{o}(d_{c}+d_{s})+\gamma(d_{c}+d_{s})2d_{h}(d_{i}+d_{o})+\gamma d_{h}

For work [Lee et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib42), Belcak and Wattenhofer, [2022](https://arxiv.org/html/2410.02675v6#bib.bib60)], they focus on different purposes from our work. And work [Lee et al., [2021](https://arxiv.org/html/2410.02675v6#bib.bib42)] assumes all input signals have the period of 1 (as stated in page 3 of its paper), which we conducted experiments on the same setting in Appendix [B.6](https://arxiv.org/html/2410.02675v6#A2.SS6 "B.6 Comparison with Directly Learning the Coefficients ‣ Appendix B Additional Experiments ‣ FAN: Fourier Analysis Networks"), and it cannot fit our periodicity modeling tasks.

### G.1 Limitation

First, we only demonstrate the effectiveness of FAN on some mainstream real-world tasks (including symbolic formula representation, time series forecasting, language modeling, image recognition, etc.), and we aim to further broaden the applicability of FAN in our future work. Second, although we have explored the generalizability of FAN and confirmed that FAN outperforms the baseline method in some real-world tasks, the boundaries of this model’s generalizability remain unknown. However, we have not yet identified specific scenarios where it performs poorly. We leave this for our future work.

[3] Title: Paper page - FAN: Fourier Analysis Networks
[3] URL Source: https://huggingface.co/papers/2410.02675
[3] Description: Join the discussion on this paper page
[3] Published Time: Wed, 10 Sep 2025 21:05:01 GMT
[3] Date: Oct 3, 2024

[![Image 2: Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

*   [Models](https://huggingface.co/models)
*   [Datasets](https://huggingface.co/datasets)
*   [Spaces](https://huggingface.co/spaces)
*   [Buckets new](https://huggingface.co/storage)
*   [Docs](https://huggingface.co/docs)
*   [Enterprise](https://huggingface.co/enterprise)
*   [Pricing](https://huggingface.co/pricing)
*   
    *   Website 
        *   [Tasks](https://huggingface.co/tasks)
        *   [HuggingChat](https://huggingface.co/chat)
        *   [Collections](https://huggingface.co/collections)
        *   [Languages](https://huggingface.co/languages)
        *   [Organizations](https://huggingface.co/organizations)

    *   Community 
        *   [Blog](https://huggingface.co/blog)
        *   [Posts](https://huggingface.co/posts)
        *   [Daily Papers](https://huggingface.co/papers)
        *   [Hardware](https://huggingface.co/hardware)
        *   [Learn](https://huggingface.co/learn)
        *   [Discord](https://huggingface.co/join/discord)
        *   [Forum](https://discuss.huggingface.co/)
        *   [GitHub](https://github.com/huggingface)

    *   Solutions 
        *   [Team & Enterprise](https://huggingface.co/enterprise)
        *   [Hugging Face PRO](https://huggingface.co/pro)
        *   [Enterprise Support](https://huggingface.co/support)
        *   [Inference Providers](https://huggingface.co/inference/models)
        *   [Inference Endpoints](https://huggingface.co/inference-endpoints)
        *   [Storage Buckets](https://huggingface.co/storage)

*   
* * *

*   [Log In](https://huggingface.co/login)
*   [Sign Up](https://huggingface.co/join)

[Papers](https://huggingface.co/papers)

arxiv:2410.02675 

Copy markdown

# FAN: Fourier Analysis Networks

Published on Oct 3, 2024

·Submitted by[![Image 3](https://cdn-avatars.huggingface.co/v1/production/uploads/64d98ef7a4839890b25eb78b/215-CSVLl81z6CAq0ECWU.jpeg) Fangyuan Yu](https://huggingface.co/Ksgk-fy)on Oct 8, 2024

[- [x] Upvote 29](https://huggingface.co/login?next=%2Fpapers%2F2410.02675)
*   [![Image 4](https://cdn-avatars.huggingface.co/v1/production/uploads/64d98ef7a4839890b25eb78b/215-CSVLl81z6CAq0ECWU.jpeg)](https://huggingface.co/Ksgk-fy "Ksgk-fy")
*   [![Image 5](https://huggingface.co/avatars/a784a51b369b197398575c3afbd5ceab.svg)](https://huggingface.co/hbkang "hbkang")
*   [![Image 6](https://huggingface.co/avatars/fb50773ac49948940eb231834ee6f2fd.svg)](https://huggingface.co/irotem98 "irotem98")
*   [![Image 7](https://cdn-avatars.huggingface.co/v1/production/uploads/62716952bcef985363db8485/zJPPo5xlwZRJdEuwYsYKp.jpeg)](https://huggingface.co/IAMJB "IAMJB")
*   [![Image 8](https://cdn-avatars.huggingface.co/v1/production/uploads/1662793811119-noauth.jpeg)](https://huggingface.co/idgmatrix "idgmatrix")
*   [![Image 9](https://huggingface.co/avatars/1c98c8be61f6580c1e4ee698fa5c0716.svg)](https://huggingface.co/learn12138 "learn12138")
*   [![Image 10](https://cdn-avatars.huggingface.co/v1/production/uploads/1635314457124-5f32b2367e583543386214d9.jpeg)](https://huggingface.co/averoo "averoo")
*   [![Image 11](https://huggingface.co/avatars/74964bfe341b865400ca36a6fc8042a0.svg)](https://huggingface.co/ccllet "ccllet")
*   +21

Authors:

![Image 12](https://huggingface.co/avatars/3acd3868232067289fad69db6c32cf87.svg)[Yihong Dong](https://huggingface.co/dongyh) ,

![Image 13](https://huggingface.co/avatars/ecc2874e612d0aed357d072427faf21a.svg)[Ge Li](https://huggingface.co/ligechina) ,

![Image 14](https://cdn-avatars.huggingface.co/v1/production/uploads/664b3aebfe822b08e62357f0/keEfUHXnkfwZ4o7588VdC.jpeg)[Yongding Tao](https://huggingface.co/YongdingTao) ,

Xue Jiang ,

![Image 15](https://huggingface.co/avatars/c7b3cf80a9a1c78802faa97b569aba4c.svg)[Kechi Zhang](https://huggingface.co/zkcpku) ,

Jia Li ,

Jing Su ,

Jun Zhang ,

![Image 16](https://huggingface.co/avatars/395710d328a824a8a84f4b78babe5809.svg)[Jingjing Xu](https://huggingface.co/Jingjingxu)

## Abstract

A new Fourier-based network architecture, FAN, efficiently models periodic phenomena with fewer parameters and demonstrates superior performance across various tasks.

 Generated by [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)

Despite the remarkable success achieved by neural networks, particularly those represented by MLP and Transformer, we reveal that they exhibit potential flaws in the modeling and reasoning of periodicity, i.e., they tend to memorize the periodic data rather than genuinely understanding the underlying principles of periodicity. However, periodicity is a crucial trait in various forms of reasoning and generalization, underpinning predictability across natural and engineered systems through recurring patterns in observations. In this paper, we propose [FAN](https://huggingface.co/papers?q=FAN), a novel network architecture based on [Fourier Analysis](https://huggingface.co/papers?q=Fourier%20Analysis), which empowers the ability to efficiently model and reason about [periodic phenomena](https://huggingface.co/papers?q=periodic%20phenomena). By introducing [Fourier Series](https://huggingface.co/papers?q=Fourier%20Series), the periodicity is naturally integrated into the structure and computational processes of the neural network, thus achieving a more accurate expression and prediction of periodic patterns. As a promising substitute to multi-layer perceptron (MLP), [FAN](https://huggingface.co/papers?q=FAN) can seamlessly replace MLP in various models with fewer parameters and FLOPs. Through extensive experiments, we demonstrate the effectiveness of [FAN](https://huggingface.co/papers?q=FAN) in modeling and reasoning about periodic functions, and the superiority and generalizability of [FAN](https://huggingface.co/papers?q=FAN) across a range of real-world tasks, including [symbolic formula representation](https://huggingface.co/papers?q=symbolic%20formula%20representation), time series forecasting, and [language modeling](https://huggingface.co/papers?q=language%20modeling).

[View arXiv page](https://arxiv.org/abs/2410.02675)[View PDF](https://arxiv.org/pdf/2410.02675)[GitHub 264 auto](https://github.com/yihongdong/fan)[Add to collection](https://huggingface.co/login?next=%2Fpapers%2F2410.02675)

### Community

![Image 17](https://cdn-avatars.huggingface.co/v1/production/uploads/64d98ef7a4839890b25eb78b/215-CSVLl81z6CAq0ECWU.jpeg)

[Ksgk-fy](https://huggingface.co/Ksgk-fy)

Paper submitter [Oct 8, 2024](https://huggingface.co/papers/2410.02675#67048d586fd3c9b6f66d5539)

Understanding periodicity with FAN

🔥

4

4

+

Reply

![Image 18](https://cdn-avatars.huggingface.co/v1/production/uploads/1674830754237-63d3e0e8ff1384ce6c5dd17d.jpeg)

[librarian-bot](https://huggingface.co/librarian-bot)

[Oct 9, 2024](https://huggingface.co/papers/2410.02675#6705ddb3ee4c6f44cfb9365d)

This is an automated message from the [Librarian Bot](https://huggingface.co/librarian-bots). I found the following papers similar to this paper.

The following papers were recommended by the Semantic Scholar API

*   [Model Comparisons: XNet Outperforms KAN](https://huggingface.co/papers/2410.02033) (2024)
*   [MLP-KAN: Unifying Deep Representation and Function Learning](https://huggingface.co/papers/2410.03027) (2024)
*   [Implicit Neural Representations with Fourier Kolmogorov-Arnold Networks](https://huggingface.co/papers/2409.09323) (2024)
*   [A Gated Residual Kolmogorov-Arnold Networks for Mixtures of Experts](https://huggingface.co/papers/2409.15161) (2024)
*   [Activation Space Selectable Kolmogorov-Arnold Networks](https://huggingface.co/papers/2408.08338) (2024)

Please give a thumbs up to this comment if you found it helpful!

If you want recommendations for any Paper on Hugging Face checkout [this](https://huggingface.co/spaces/librarian-bots/recommend_similar_papers) Space

You can directly ask Librarian Bot for paper recommendations by tagging it in a comment: `@librarian-bot recommend`

Reply

![Image 19](https://cdn-avatars.huggingface.co/v1/production/uploads/63c1eead726f62e411fb1b55/smCV00dkY0phGlUYLg1zJ.jpeg)

[glamprou](https://huggingface.co/glamprou)

[Oct 9, 2024](https://huggingface.co/papers/2410.02675#670685763985eccceab8aa2f)

Great work, easy to read paper and implement. Thank You! I've made my own implementation [jlamprou/Fourier-Analysis-Networks-FAN](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN).

*   [![Image 20](https://huggingface.co/avatars/3acd3868232067289fad69db6c32cf87.svg)](https://huggingface.co/dongyh "dongyh")
*   1 reply

·

![Image 21](https://huggingface.co/avatars/3acd3868232067289fad69db6c32cf87.svg)

[dongyh](https://huggingface.co/dongyh)

Paper author [Oct 13, 2024](https://huggingface.co/papers/2410.02675#670c0d814d6a12dc2cd65c3a)

Thank you very much for your attention!

![Image 22](https://huggingface.co/avatars/e0d61335ff146de947d5742f80ff5220.svg)

[jgcb00](https://huggingface.co/jgcb00)

[Oct 17, 2024](https://huggingface.co/papers/2410.02675#6711404f95470ee597e1fa94)

•

[edited Oct 17, 2024](https://huggingface.co/papers/2410.02675#6711404f95470ee597e1fa94 "Edited by jgcb00")

Hi there seems to be an error in the figure2, it should be like that if I understand well

[![Image 23: image.png](https://cdn-uploads.huggingface.co/production/uploads/64f0a2ac5b9c8cdb17786783/_YOv4u4S4lq4jQ28V2PKF.png)](https://cdn-uploads.huggingface.co/production/uploads/64f0a2ac5b9c8cdb17786783/_YOv4u4S4lq4jQ28V2PKF.png)

Am I right ?

Otherwise, it's very interesting, thanks

*   [![Image 24](https://huggingface.co/avatars/3acd3868232067289fad69db6c32cf87.svg)](https://huggingface.co/dongyh "dongyh")
*   1 reply

·

![Image 25](https://huggingface.co/avatars/3acd3868232067289fad69db6c32cf87.svg)

[dongyh](https://huggingface.co/dongyh)

Paper author [Oct 17, 2024](https://huggingface.co/papers/2410.02675#67114969379537de06e3d1b7)

Yes, you are right! Thank you very much for pointing out this mistake, we will revise it in our subsequent version.

Edit Preview

Upload images, audio, and videos by dragging in the text input, pasting, or clicking here.

Tap or paste here to upload images

 

 Comment
·[Sign up](https://huggingface.co/join?next=%2Fpapers%2F2410.02675) or [log in](https://huggingface.co/login?next=%2Fpapers%2F2410.02675) to comment

[- [x] Upvote 29](https://huggingface.co/login?next=%2Fpapers%2F2410.02675)
*   [![Image 26](https://cdn-avatars.huggingface.co/v1/production/uploads/64d98ef7a4839890b25eb78b/215-CSVLl81z6CAq0ECWU.jpeg)](https://huggingface.co/Ksgk-fy "Ksgk-fy")
*   [![Image 27](https://huggingface.co/avatars/a784a51b369b197398575c3afbd5ceab.svg)](https://huggingface.co/hbkang "hbkang")
*   [![Image 28](https://huggingface.co/avatars/fb50773ac49948940eb231834ee6f2fd.svg)](https://huggingface.co/irotem98 "irotem98")
*   [![Image 29](https://cdn-avatars.huggingface.co/v1/production/uploads/62716952bcef985363db8485/zJPPo5xlwZRJdEuwYsYKp.jpeg)](https://huggingface.co/IAMJB "IAMJB")
*   [![Image 30](https://cdn-avatars.huggingface.co/v1/production/uploads/1662793811119-noauth.jpeg)](https://huggingface.co/idgmatrix "idgmatrix")
*   [![Image 31](https://huggingface.co/avatars/1c98c8be61f6580c1e4ee698fa5c0716.svg)](https://huggingface.co/learn12138 "learn12138")
*   [![Image 32](https://cdn-avatars.huggingface.co/v1/production/uploads/1635314457124-5f32b2367e583543386214d9.jpeg)](https://huggingface.co/averoo "averoo")
*   [![Image 33](https://huggingface.co/avatars/74964bfe341b865400ca36a6fc8042a0.svg)](https://huggingface.co/ccllet "ccllet")
*   [![Image 34](https://huggingface.co/avatars/c12f4cb6dc1ff0010edb3ef4cfcccd7c.svg)](https://huggingface.co/Inversta "Inversta")
*   [![Image 35](https://huggingface.co/avatars/f1dec2f815bc270a6c5d32c4cbec8017.svg)](https://huggingface.co/lxc0422 "lxc0422")
*   [![Image 36](https://huggingface.co/avatars/e9e2393d41287c46d13e42a96a38fa9c.svg)](https://huggingface.co/dj3m "dj3m")
*   [![Image 37](https://cdn-avatars.huggingface.co/v1/production/uploads/620783f24e28382272337ba4/zkUveQPNiDfYjgGhuFErj.jpeg)](https://huggingface.co/Tommy930 "Tommy930")
*   +17

Get this paper in your agent:

`hf papers read 2410.02675`

Don't have the latest CLI?

`curl -LsSf https://hf.co/cli/install.sh | bash`

## Models citing this paper 1

[![Image 38](https://huggingface.co/avatars/3acd3868232067289fad69db6c32cf87.svg) #### dongyh/FANformer-1B Text Generation • 1B•Updated Mar 30, 2025• 18• 5](https://huggingface.co/dongyh/FANformer-1B)
## Datasets citing this paper 0

No dataset linking this paper

Cite arxiv.org/abs/2410.02675 in a dataset README.md to link it from this page.

### Spaces citing this paper 0

No Space linking this paper

Cite arxiv.org/abs/2410.02675 in a Space README.md to link it from this page.

## Collections including this paper 9

#### [ByteDance Papers Collection ByteDance papers collection•142 items•Updated 1 day ago• 36](https://huggingface.co/collections/Presidentlin/bytedance-papers)

#### [Good Papers Collection 149 items•Updated Dec 26, 2025• 8](https://huggingface.co/collections/steveyin/good-papers)

#### [Cognition Collection Perception and abstraction. Each modality is tokenized and embedded into vectors for model to comprehend. •201 items•Updated 23 days ago• 6](https://huggingface.co/collections/Ksgk-fy/cognition)

#### [interesting architecture Collection 47 items•Updated Jun 27• 5](https://huggingface.co/collections/hbkang/interesting-architecture)

[Browse 9 collections that include this paper](https://huggingface.co/collections?paper=2410.02675)

 System theme

Company

[TOS](https://huggingface.co/terms-of-service)[Privacy](https://huggingface.co/privacy)[About](https://huggingface.co/huggingface)[Careers](https://apply.workable.com/huggingface/)[](https://huggingface.co/)

Website

[Models](https://huggingface.co/models)[Datasets](https://huggingface.co/datasets)[Spaces](https://huggingface.co/spaces)[Pricing](https://huggingface.co/pricing)[Docs](https://huggingface.co/docs)

[4] Title: GitHub - jlamprou/Fourier-Analysis-Networks-FAN: A pytorch implementation of Fourier Analysis Networks (FAN)
[4] URL Source: https://github.com/jlamprou/Fourier-Analysis-Networks-FAN
[4] Description: A pytorch implementation of Fourier Analysis Networks (FAN) - jlamprou/Fourier-Analysis-Networks-FAN

This repository contains an implementation of Fourier Analysis Networks (FAN) as described in the paper ["FAN: Fourier Analysis Networks" by Yihong Dong, Ge Li, et al](https://arxiv.org/abs/2410.02675)

## Table of Contents

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#table-of-contents)
*   [Introduction](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#introduction)
*   [Requirements](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#requirements)
*   [Usage](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#usage)
*   [Implementation Details](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#implementation-details)
*   [Experiments](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#experiments)
*   [Contributing](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#contributing)
*   [Citation](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#citation)
*   [License](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#license)

## Introduction

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#introduction)
Fourier Analysis Networks (FAN) is a novel neural network architecture designed to efficiently model and reason about periodic phenomena. By introducing Fourier Series into the structure and computational processes of neural networks, FAN achieves more accurate expression and prediction of periodic patterns.

Key features of FAN:

*   Enhanced ability to model periodicity
*   Fewer parameters and FLOPs compared to traditional MLPs
*   Improved performance on various tasks, including symbolic formula representation, time series forecasting, and language modeling

## Requirements

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#requirements)
Just PyTorch...

## Usage

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#usage)
Here's a basic example of how to use the FAN layer:

import torch
from fan import FANLayer

# Initialize a FAN layer
input_dim = 64
output_dim = 128
fan_layer = FANLayer(input_dim, output_dim)

# Create a random input tensor
x = torch.randn(32, input_dim)  # batch size of 32

# Pass the input through the FAN layer
output = fan_layer(x)

print(output.shape)  # Should be torch.Size([32, 128])

## Implementation Details

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#implementation-details)
The core of this implementation is the `FANLayer` class. This layer implements the FAN architecture as described in the paper, including:

*   Initialization of learnable parameters W_p, W_p̄, and B_p̄
*   Forward pass computation of cos(W_p x), sin(W_p x), and σ(B_p̄ + W_p̄ x)
*   Concatenation of the computed terms

The implementation also includes variants such as FAN (Gated) and models that combine FAN with other architectures like Transformer.

## Experiments

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#experiments)
Soon we will test FAN on downstream tasks.

## Contributing

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#contributing)
We welcome contributions to improve the implementation or add new features. Please feel free to submit pull requests or open issues for any bugs or feature requests.

## Citation

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#citation)
If you use this implementation in your research, please cite the original paper:

@article{dong2024fan,
  title={FAN: Fourier Analysis Networks},
  author={Yihong Dong and Ge Li and Yongding Tao and Xue Jiang and Kechi Zhang and Jia Li and Jing Su and Jun Zhang and Jingjing Xu},
  journal={arXiv preprint arXiv:2410.02675},
  year={2024}
}

## License

[](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN#license)
This project is licensed under the MIT License - see the [LICENSE](https://github.com/jlamprou/Fourier-Analysis-Networks-FAN/blob/main/LICENSE) file for details.

[5] Title: Just a moment...
[5] URL Source: https://www.researchgate.net/publication/384630742_FAN_Fourier_Analysis_Networks
[5] Description: 
[5] Date: Oct 15, 2024

## Security check required

We've detected unusual activity from your network. To continue, complete the security check below.

Ray ID: a29a79147faa5001

Client IP: 2600:1900:0:2104::1601

© 2008-2026 ResearchGate GmbH. All rights reserved.

[6] Title: \modelname: Fourier Analysis Networks
[6] URL Source: https://arxiv.org/html/2410.02675v4/
[6] Description: 
[6] Published Time: Thu, 03 Apr 2025 00:26:13 GMT

Yihong Dong 1, Ge Li 1 1 1 footnotemark: 1, Yongding Tao 1, Xue Jiang 1, Kechi Zhang 1, Jia Li ♂1, 

Jinliang Deng 2, Jing Su 3, Jun Zhang 3, Jingjing Xu 3

1 School of Computer Science, Peking University 

2 The Hong Kong University of Science and Technology 3 ByteDance 

dongyh@stu.pku.edu.cn, lige@pku.edu.cn

###### Abstract

Despite the remarkable successes of general-purpose neural networks, such as MLPs and Transformers, we find that they exhibit notable shortcomings in modeling and reasoning about periodic phenomena, achieving only marginal performance within the training domain and failing to generalize effectively to out-of-domain (OOD) scenarios. Periodicity is ubiquitous throughout nature and science. Therefore, neural networks should be equipped with the essential ability to model and handle periodicity. In this work, we propose FAN, a novel general-purpose neural network that offers broad applicability similar to MLP while effectively addressing periodicity modeling challenges. Periodicity is naturally integrated into FAN’s structure and computational processes by introducing the Fourier Principle. Unlike existing Fourier-based networks, which possess particular periodicity modeling abilities but are typically designed for specific tasks, our approach maintains the general-purpose modeling capability. Therefore, FAN can seamlessly replace MLP in various model architectures with fewer parameters and FLOPs. Through extensive experiments, we demonstrate the superiority of FAN in periodicity modeling tasks and the effectiveness and generalizability of FAN across a range of real-world tasks, e.g., symbolic formula representation, time series forecasting, language modeling, and image recognition. ††footnotetext: This work was supported by a cooperation project between Peking University and ByteDance Company. During this time, Yihong was also an intern at ByteDance.‡‡footnotetext: The code is available at [https://github.com/YihongDong/FAN](https://github.com/YihongDong/FAN)

## 1 Introduction

The flourishing of modern machine learning and artificial intelligence is inextricably linked to the revolutionary advancements in the foundational architecture of general-purpose neural networks. For instance, multi-layer perceptron (MLP)(Rosenblatt, [1958](https://arxiv.org/html/2410.02675v4#bib.bib36); Haykin, [1998](https://arxiv.org/html/2410.02675v4#bib.bib10)) plays a pivotal role in laying the groundwork for current deep learning models, with its expressive power guaranteed by the universal approximation theorem(Hornik et al., [1989](https://arxiv.org/html/2410.02675v4#bib.bib13)). Recent claims about the impressive performance of large models on various tasks are typically supported by Transformer architecture(Vaswani et al., [2017](https://arxiv.org/html/2410.02675v4#bib.bib48); Touvron et al., [2023](https://arxiv.org/html/2410.02675v4#bib.bib45); OpenAI, [2023](https://arxiv.org/html/2410.02675v4#bib.bib31)). In this context, the community’s enthusiasm for research on neural networks has never diminished. Some emerged neural networks demonstrate notable capabilities in specific fields(Gu & Dao, [2023](https://arxiv.org/html/2410.02675v4#bib.bib7); Liu et al., [2024](https://arxiv.org/html/2410.02675v4#bib.bib24)), sparking widespread discussion within the community.

Beneath the surface of apparent prosperity, we uncover a critical issue that remains in existing general-purpose neural networks: they struggle to model the periodicity from data, especially in OOD scenarios. We showcase this issue through an empirical study as illustrated in Figure [1](https://arxiv.org/html/2410.02675v4#S1.F1 "Figure 1 ‣ 1 Introduction ‣ \modelname: Fourier Analysis Networks"). The results indicate that existing neural networks, including MLP (Rosenblatt, [1958](https://arxiv.org/html/2410.02675v4#bib.bib36)), KAN (Liu et al., [2024](https://arxiv.org/html/2410.02675v4#bib.bib24)), and Transformer (Vaswani et al., [2017](https://arxiv.org/html/2410.02675v4#bib.bib48)), face difficulties in fitting periodic functions, even on a simple sine function. Although they demonstrate some proficiency in interpolation within the domain of training data, they tend to falter when faced with extrapolation challenges of test data. This signifies that their generalization capacity is primarily dictated by the scale and diversity of the training data, rather than by the learned principles of periodicity to perform reasoning.

![Image 1: Refer to caption](https://arxiv.org/html/2410.02675v4/x1.png)

Figure 1: The performance of different neural networks within and outside the domain of their training data for the sine function, where x is a scalar variable. 

Periodicity is an essential characteristic in various forms of reasoning and generalization, as it provides a basis for predictability in many natural and engineered systems by leveraging recurring patterns in observations. Besides periodic phenomena, non-periodic phenomena can also be contextualized or explained within some larger or more macro-periodic framework. Although some Fourier-based networks exhibit particular periodic modeling abilities, they are primarily tailored for specific tasks (Silvescu, [1999](https://arxiv.org/html/2410.02675v4#bib.bib38); Liu, [2013](https://arxiv.org/html/2410.02675v4#bib.bib23)) and do not work well as the networks deepen (Liu et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib25)), which limits their applicability to the general task such as language modeling (Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib47)). However, our goal is to exploit periodicity to benefit a broader range of tasks including language modeling. To achieve this, we aim to develop a general-purpose neural network that accommodates modeling and reasoning capabilities for periodicity.

In this paper, we propose a Fourier Analysis Network (\modelname), a novel general-purpose neural network built upon the principle of Fourier Analysis. By leveraging the power of Fourier Series, we enable the neural network to model periodic patterns and extrapolate beyond them, offering the network a way to model the general principles from the data. \modelname follows two core principles, the first ensures that its periodic modeling capacity scales with network depth, while the second guarantees periodic modeling available throughout the network. As a result, \modelname exhibits exceptional capabilities in periodicity modeling, while maintaining broad applicability to the general task, which holds great potential as a substitute for MLP, with fewer parameters and FLOPs.

To verify the effectiveness of \modelname, we conduct extensive experiments from two main aspects: periodicity modeling and application of real-world tasks. 1) For periodicity modeling, \modelname achieves significant improvements in fitting both basic and complex periodic functions, compared to existing neural networks (including MLP, KAN, and Transformer), particularly in OOD scenarios. 2) \modelname shows superior performance in various real-world tasks, such as symbolic formula representation, time series forecasting, language modeling, and image recognition. Using \modelname outperforms the representative models in various tasks, including MLP, Transformer, KAN, LSTM, Mamba, and CNN. Notably, compared to Fourier-based networks, \modelname accommodates both periodicity modeling and language modeling tasks well. The advantageous characteristics and promising results indicate that FAN has the potential to become a basic component for building fundamental large models.

## 2 Preliminary Knowledge

Fourier Analysis(Stein & Weiss, [1971](https://arxiv.org/html/2410.02675v4#bib.bib41); Duoandikoetxea, [2024](https://arxiv.org/html/2410.02675v4#bib.bib5)) is a mathematical framework that decomposes functions into their constituent frequencies, revealing the underlying periodic structures within complex functions. At the heart of this analysis lies Fourier Series(Tolstov, [2012](https://arxiv.org/html/2410.02675v4#bib.bib44)), which expresses a periodic function as an infinite sum of sine and cosine terms. Mathematically, for a function f(x), its Fourier Series expansion can be represented as:

f(x)=a_{0}+\sum_{n=1}^{\infty}\left(a_{n}\cos\left(\frac{2\pi nx}{T}\right)+b_%
{n}\sin\left(\frac{2\pi nx}{T}\right)\right),(1)

where T is the period of the function, and the coefficients a_{n} and b_{n} are determined by integrating the function over one period:

a_{n}=\frac{1}{T}\int_{0}^{T}f(x)\cos\left(\frac{2\pi nx}{T}\right)\,dx,\quad b%
_{n}=\frac{1}{T}\int_{0}^{T}f(x)\sin\left(\frac{2\pi nx}{T}\right)\,dx.(2)

The power of Fourier Series lies in its ability to represent a wide variety of functions, including non-periodic functions through periodic extensions, enabling the extraction of frequency components. Building on this mathematical foundation, \modelname aims to embed the periodic characteristics directly into network architecture, enhancing generalization capabilities and performance on various tasks, particularly in scenarios requiring the identification of patterns and regularities.

## 3 Fourier Analysis Network (\modelname)

In this section, we first construct a naive neural network modeled by the formula of Fourier Series. Then, by modifying and improving it, we design \modelname adhering to two core principles. Finally, we discuss the difference between the \modelname layer and MLP layer.

Consider a task involving input-output pairs \{x_{i},y_{i}\}, with the objective of identifying a function f(x):\mathbb{R}^{d_{x}}\rightarrow\mathbb{R}^{d_{y}} that approximates the relationship such that y_{i}\approx f(x_{i}) for all x_{i}, where d_{x} and d_{y} denote the dimensions of x and y, respectively. We first construct a shallow neural network f_{\text{S}}(x) that represents Fourier Series expansion of the function, specifically \mathcal{F}\{f(x)\}, as described in Eq. ([1](https://arxiv.org/html/2410.02675v4#S2.E1 "In 2 Preliminary Knowledge ‣ \modelname: Fourier Analysis Networks")), we can express f_{\text{S}}(x) as follows:

\displaystyle f_{\text{S}}(x)\displaystyle\triangleq a_{0}+\sum_{n=1}^{N}\left(a_{n}\cos\left(\frac{2\pi nx%
}{T}\right)+b_{n}\sin\left(\frac{2\pi nx}{T}\right)\right),(3)
\displaystyle\mathop{=}\limits^{(\text{I})}a_{0}+\sum_{n=1}^{N}\left(w^{c}_{n}%
\cos\left(w^{\text{in}}_{n}x\right)+w^{s}_{n}\sin\left(w^{\text{in}}_{n}x%
\right)\right),
\displaystyle\mathop{=}\limits^{(\text{II})}B+[w^{c}_{1},w^{c}_{2},\cdots,w^{c%
}_{n}]\cos([w^{\text{in}}_{1}||w^{\text{in}}_{2}||\cdots||w^{\text{in}}_{n}]x)
\displaystyle\quad+[w^{s}_{1},w^{s}_{2},\cdots,w^{s}_{n}]\sin([w^{\text{in}}_{%
1}||w^{\text{in}}_{2}||\cdots||w^{\text{in}}_{n}]x)
\displaystyle=B+W_{c}\cos(W_{\text{in}}x)+W_{s}\sin(W_{\text{in}}x),
\displaystyle\mathop{=}\limits^{(\text{III})}B+W_{\text{out}}[\cos(W_{\text{in%
}}x)||\sin(W_{\text{in}}x)],

where B\in\mathbb{R}^{d_{y}},W_{\text{in}}\in\mathbb{R}^{N\times d_{x}}, and W_{\text{out}}\in\mathbb{R}^{d_{y}\times 2N} are learnable parameters, (\text{I}) follows that the computation of a_{n} and b_{n} computed via Eq. ([2](https://arxiv.org/html/2410.02675v4#S2.E2 "In 2 Preliminary Knowledge ‣ \modelname: Fourier Analysis Networks")) is definite integral, (\text{II}) and (\text{III}) follows the equivalence of the matrix operations, [\cdot||\cdot] and [\cdot,\cdot] denotes the concatenation along the first and second dimension, respectively.

![Image 2: Refer to caption](https://arxiv.org/html/2410.02675v4/x2.png)

Figure 2: Illustrations of \modelname layer \phi(x) vs. MLP layer \Phi(x).

To fully leverage the advantages of deep learning, we can stack the aforementioned network f_{\text{S}}(x) to form a deep network f_{\text{D}}(x), where the i-th layer, denoted as l_{i}(x), retains the same structural design as f_{\text{S}}(x). Therefore, f_{\text{D}}(x) can be formulated as:

f_{\text{D}}(x)=l_{L}\circ l_{L-1}\circ\cdots\circ l_{1}\circ x,(4)

where l_{1}\circ x denotes the application of the left function l_{1} to the right input x, that is l_{1}(x). However, we discover that the direct stacking of f_{\text{S}}(x) results in the primary parameters of the network f_{\text{D}}(x) focusing on learning the angular frequency (\omega_{n}=\frac{2\pi n}{T}), thereby neglecting the learning of the Fourier coefficients (a_{n} and b_{n}), as follows:

\displaystyle f_{\text{D}}(x)=l_{L}(l_{L-1}\circ l_{L-2}\circ\cdots\circ l_{1}%
\circ x)(5)
\displaystyle=B^{L}+W^{L}_{\text{out}}[\cos(W^{L}_{\text{in}}(l_{1:L}\circ x)|%
|\sin(W^{L}_{\text{in}}(l_{1:L}\circ x))]

where l_{1:L}\circ x is defined as l_{L-1}\circ l_{L-2}\circ\cdots\circ l_{1}\circ x, W^{L}_{\text{in}}(l_{1:L}\circ x) is used to approximate the angular frequencies, and W^{L}_{\text{out}} is used to approximate the Fourier coefficients. We can find that the capacity of f_{\text{D}}(x) to fit the Fourier coefficients is independent of the depth of f_{\text{D}}(x), which is an undesirable outcome. It will limit the network’s representation ability, hindering to address the complex tasks.

To this end, we design \modelname based on the following principles: 1) the capacity of \modelname to represent the Fourier coefficients should be positively correlated to its depth; 2) the output of any hidden layer can be employed to model periodicity using Fourier Series through the subsequent layers. The first one enhances the expressive power of \modelname for periodicity modeling by leveraging its depth, while the second one ensures that the features of \modelname’s intermediate layers are available to perform periodicity modeling.

Suppose we decouple f_{\text{S}}(x) as follows:

f_{\text{S}}(x)=f_{out}\circ f_{in}\circ x,(6)

where

\displaystyle f_{in}(x)=[\cos(W_{\text{in}}x)||\sin(W_{\text{in}}x)],(7)
\displaystyle f_{out}(x)=B+W_{\text{out}}x.(8)

To satisfy both principles, the inputs of the intermediate layers in \modelname necessitate to employ f_{in} and f_{out} simultaneously, rather than applying them sequentially.

Finally, \modelname is designed on this basis, with the \modelname layer \phi(x) defined as below:

\phi(x)\triangleq[\cos(W_{p}x)||\sin(W_{p}x)||\sigma(B_{\bar{p}}+W_{\bar{p}}x)],(9)

where W_{p}\in\mathbb{R}^{d_{x}\times d_{p}},W_{\bar{p}}\in\mathbb{R}^{d_{x}\times d%
_{\bar{p}}}, and B_{\bar{p}}\in\mathbb{R}^{d_{\bar{p}}} are learnable parameters (with the hyperparameters d_{p} and d_{\bar{p}} indicating the first dimension of W_{p} and W_{\bar{p}}, respectively), the layer output \phi(x)\in\mathbb{R}^{2d_{p}+d_{\bar{p}}}, and \sigma denotes the activation function. Under this definition, the MLP layer can be regarded as a special form of Eq. ([9](https://arxiv.org/html/2410.02675v4#S3.E9 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks")), when W_{p} are learned to be zero metrics, which provide a way for FAN to maintain general-purpose modeling abilities as MLP.

Table 1: Comparison of \modelname layer and MLP layer, where d_{\text{p}} is a hyperparameter of \modelname layer and defaults to \frac{1}{4}d_{\text{output}} in this paper, d_{\text{input}} and d_{\text{output}} denote the input and output dimensions of the neural network layer, respectively. In our evaluation, the floating point of operations (FLOPs) for any arithmetic operations are considered as 1, and for Boolean operations as 0.

MLP Layer\modelname layer
Formula\Phi(x)=\sigma(B_{m}+W_{m}x)\phi(x)=[\cos(W_{p}x)||\sin(W_{p}x)||\sigma(B_{\bar{p}}+W_{\bar{p}}x)]
\hdashline Num of Params(d_{\text{input}}\times d_{\text{output}})+d_{\text{output}}(1-\frac{d_{p}}{d_{\text{output}}})\times((d_{\text{input}}\times d_{\text{%
output}})+d_{\text{output}})
\hdashline FLOPs 2\times(d_{\text{input}}\times d_{\text{output}})+\text{FLOPs}_{\text{non-linear}}\times d_{\text{output}}(1-\frac{d_{p}}{d_{\text{output}}})\times 2\times(d_{\text{input}}\times d_{%
\text{output}})+\text{FLOPs}_{\text{non-linear}}\times d_{\text{output}}

The entire \modelname is defined as the stacking of the \modelname layer \phi(x) as follows:

\text{FAN}(x)=\phi_{L}\circ\phi_{L-1}\circ\cdots\circ\phi_{1}\circ x,(10)

where

\small{\phi_{l}(x)=\left\{\begin{array}[]{ll}[\cos(W^{l}_{p}x)||\sin(W^{l}_{p}%
x)||\sigma(B^{l}_{\bar{p}}+W^{l}_{\bar{p}}x)],&\text{if }l<L,\\
B^{L}+W^{L}x,&\text{if }l=L,\end{array}\right.}(11)

#### The difference between FAN and MLP.

The illustrations of \modelname layer \phi(x) vs. MLP layer \Phi(x) are shown in Figure [2](https://arxiv.org/html/2410.02675v4#S3.F2 "Figure 2 ‣ 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks"). Note that the \modelname layer \phi(x) computed via Eq. ([9](https://arxiv.org/html/2410.02675v4#S3.E9 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks")) can seamlessly replace the MLP layer \Phi(x) computed via Eq. ([12](https://arxiv.org/html/2410.02675v4#A1.E12 "In Appendix A MLP ‣ \modelname: Fourier Analysis Networks")) in various models with fewer parameters and FLOPs, achieved by sharing the parameters and computation of Sin and Cos parts. The number of parameters and FLOPs of the \modelname layer compared to the MLP layer are presented in Table [1](https://arxiv.org/html/2410.02675v4#S3.T1 "Table 1 ‣ 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks"). The reduction ratio of parameters and FLOPs is about \frac{d_{p}}{d_{\text{output}}}, which is set to \frac{1}{4} by default in this paper.

## 4 Experiments

In this section, we first introduce the baselines of our experiments. Second, we verify the superiority of \modelname in periodicity modeling tasks (Section [4.1](https://arxiv.org/html/2410.02675v4#S4.SS1 "4.1 Periodicity Modeling ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks")). Third, we demonstrate the effectiveness and generalizability of \modelname across a range of real-world tasks (Section [4.2](https://arxiv.org/html/2410.02675v4#S4.SS2 "4.2 Application of Real-world Task ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks")). Finally, we conduct further analysis of FAN (Section [4.3](https://arxiv.org/html/2410.02675v4#S4.SS3 "4.3 Further Analysis of FAN ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks")), including comparisons with Fourier-based networks, running time, etc. See Appendix [B](https://arxiv.org/html/2410.02675v4#A2 "Appendix B Additional Experiments ‣ \modelname: Fourier Analysis Networks") for more experiments and the experimental details can be found in Appendix [C](https://arxiv.org/html/2410.02675v4#A3 "Appendix C Experimental Details ‣ \modelname: Fourier Analysis Networks").

Baselines. In our experiments, we mainly compare \modelname with the following baselines: 1) MLP(Rosenblatt, [1958](https://arxiv.org/html/2410.02675v4#bib.bib36)), 2) Transformer(Vaswani et al., [2017](https://arxiv.org/html/2410.02675v4#bib.bib48)), 3) KAN(Liu et al., [2024](https://arxiv.org/html/2410.02675v4#bib.bib24)), 4) LSTM(Hochreiter & Schmidhuber, [1997](https://arxiv.org/html/2410.02675v4#bib.bib12)), 5) Mamba(Gu & Dao, [2023](https://arxiv.org/html/2410.02675v4#bib.bib7)), 6) CNN(LeCun et al., [1998](https://arxiv.org/html/2410.02675v4#bib.bib17)). In analysis, we also compare \modelname with Fourier-based networks(Silvescu, [1999](https://arxiv.org/html/2410.02675v4#bib.bib38); Liu, [2013](https://arxiv.org/html/2410.02675v4#bib.bib23); Li et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib21)). Details of the baselines are given in Appendix [F](https://arxiv.org/html/2410.02675v4#A6 "Appendix F More Details of Baselines ‣ \modelname: Fourier Analysis Networks"). Moreover, we further include the following variants of \modelname into our comparisons: I) FAN(Gated): a variant of \modelname that adds gates to control the tendency of the layer, with the formula defined as \phi_{g}(x)=[g\cdot\cos(W_{p}x)||g\cdot\sin(W_{p}x)||(1-g)\cdot\sigma(B_{\bar{%
p}}+W_{\bar{p}}x)], where g is a learnable parameter. II) Transformer with FAN and Transformer with FAN(Gated): we replace each MLP layer in Transformer with the \modelname layer computed via Eq. ([9](https://arxiv.org/html/2410.02675v4#S3.E9 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks")) and the layer of \modelname(Gated), respectively. III) CNN with FAN: similarly, we replace each MLP layer in CNN with the \modelname layer.

![Image 3: Refer to caption](https://arxiv.org/html/2410.02675v4/x3.png)

![Image 4: Refer to caption](https://arxiv.org/html/2410.02675v4/x4.png)

Figure 3: The performance of \modelname in periodicity modeling compared to MLP, KAN, and Transformer, where the green line represents the test data within the domain of training data, while the blue line represents the test data outside the domain of training data.

![Image 5: Refer to caption](https://arxiv.org/html/2410.02675v4/x5.png)

Figure 4: Comparison of training and test losses for different models on the tasks of learning complex periodic functions.

### 4.1 Periodicity Modeling

#### Setup.

In periodic modeling tasks, we select periodic functions with practical significance and compare the models’ performance in learning the underlying principles of periodicity. Specifically, we generate data from periodic functions over a large domain, using a portion of this domain as training data and the entire domain as test data, i.e., a part of test data would be out of the domain of training data. We compare \modelname and its variant \modelname(Gated), with MLP, KAN, and Transformer. The input of this task is scalar.

Results. Figure [3](https://arxiv.org/html/2410.02675v4#S4.F3 "Figure 3 ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"), along with Figure [7](https://arxiv.org/html/2410.02675v4#A1.F7 "Figure 7 ‣ Appendix A MLP ‣ \modelname: Fourier Analysis Networks") of Appendix, show the performance of \modelname and other baselines in periodicity modeling. The results indicate that existing neural networks, including MLP, KAN, and Transformers, exhibit notable deficiencies in their ability to model periodicity. Although they attempt to fit these periodic functions, their ability limits their performance in modeling a large domain of periodicity, including the test data within and outside the domain of the training data. In contrast, \modelname significantly outperforms baselines in all these tasks of periodicity modeling. Moreover, \modelname performs exceptionally well on the test data both within and outside the domain, indicating that our specialized design of FAN can effectively model and understand periodicity rather than merely memorize the training data.

We also compare the training process of different models on the tasks of learning complex periodic functions, as shown in Figure [4](https://arxiv.org/html/2410.02675v4#S4.F4 "Figure 4 ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"), which leads to the following findings. 1) \modelname far exceeds the other baselines in both convergence speed and final effects. 2) \modelname(Gated) often achieves faster convergence than \modelname, but the final performance remains comparable. 3) Although the baselines show stabilization or gradual reductions in training loss as the number of epochs increases, their modeling may have diverged considerably from the distribution of the test data, resulting in a sharp increase in test loss. This phenomenon further demonstrates the shortcomings of these models in capturing periodicity.

### 4.2 Application of Real-world Task

1) Symbolic Formula Representation is a common task in both mathematics and physics. We follow the experiments conducted in KAN’s paper (Liu et al., [2024](https://arxiv.org/html/2410.02675v4#bib.bib24)), adhering to the same tasks, data, hyperparameters, and baselines. In addition to the original baselines, we also include Transformer for comparison in this task.

![Image 6: Refer to caption](https://arxiv.org/html/2410.02675v4/x6.png)

Figure 5: Comparisons of \modelname with the baselines, including MLP, KAN, and Transformer, across varying numbers of parameters on symbolic formula representation tasks.

Results. Figure [5](https://arxiv.org/html/2410.02675v4#S4.F5 "Figure 5 ‣ 4.2 Application of Real-world Task ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks") shows the performance of different models applied to common functions in mathematics and physics. We can observe that while KAN remains competitive with \modelname when the number of parameters is small, its performance declines clearly as the number of parameters increases, which exhibits a U-shaped trend (Liu et al., [2024](https://arxiv.org/html/2410.02675v4#bib.bib24)). In contrast, as the number of parameters becomes large, \modelname consistently outperforms the other baselines, including MLP, KAN, and Transformer, in fitting these functions, despite many of these functions being only partially periodic or even implicitly periodic. This may be attributed to FAN’s ability to capture and model both periodic and non-periodic features and the advantages of fewer parameters. These results indicate that although \modelname enhances its ability to model periodicity, it does not compromise its capacity to fit non-periodic functions.

2) Time Series Forecasting plays a critical role in various real-world applications. We employ four public datasets of this task to assess the model performance on time series forecasting, including Weather (Wu et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib50)), Exchange (Lai et al., [2018](https://arxiv.org/html/2410.02675v4#bib.bib16)), Traffic (Wu et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib50)), and ETTh (Zhou et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib53)) datasets. For each dataset, we input 96 previous time steps and forecast the subsequent time steps of {96, 192, 336, 720}. In this task, we choose the sequence models as baselines, including LSTM, Mamba, and Transformer.

Table 2: Average performance on different public datasets and output lengths in time series forecasting tasks, where Input Length = 96 and the bold value indicates the best performance.

Model Num of Params Average
MSE \downarrow MAE \downarrow
LSTM 12.51M 1.083 0.726
Mamba 12.69M 1.002 0.668
Transformer 12.12M 0.994 0.689
\hdashline w/ \modelname(Gated)11.07M 0.845 0.637
w/ \modelname 11.06M 0.839 0.631
Improvements\downarrow 1.06M\downarrow 15.6%\downarrow 8.4%

Results. As shown in Table [2](https://arxiv.org/html/2410.02675v4#S4.T2 "Table 2 ‣ 4.2 Application of Real-world Task ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks") (See Table [7](https://arxiv.org/html/2410.02675v4#A1.T7 "Table 7 ‣ Appendix A MLP ‣ \modelname: Fourier Analysis Networks") of Appendix for complete results), we compare the performance of Transformer with \modelname and other baselines for time series forecasting tasks. The results indicate that Transformer with \modelname outperforms other representative sequence models in these tasks. The improvements of Transformer with \modelname and \modelname(Gated) over standard Transformer are notable, with average relative improvements ranging from 15.0% to 15.6% for MSE and from 7.6% to 8.4% for MAE. It suggests that incorporating explicit periodic pattern encoding within neural networks improves time series forecasting performance in real-world applications.

3) Language Modeling is a fundamental task in natural language processing. We conduct language modeling using the SST-2 (Socher et al., [2013](https://arxiv.org/html/2410.02675v4#bib.bib40)) dataset and evaluate the model’s performance on its test set, as well as on the related datasets such as IMDB (Maas et al., [2011](https://arxiv.org/html/2410.02675v4#bib.bib27)), Sentiment140 (Sahni et al., [2017](https://arxiv.org/html/2410.02675v4#bib.bib37)), and Amazon Reviews (Linden et al., [2003](https://arxiv.org/html/2410.02675v4#bib.bib22)). These four classic datasets all belong to the field of sentiment analysis. The comparisons are between Transformer with \modelname and \modelname(Gated), along with the classic sequence models, including LSTM, Mamba, and Transformer.

Table 3: Performance of different sequence models on language modeling tasks, where the models are trained on the training set of SST-2 and evaluated on the other datasets, the bold value indicates the best performance on each column, the bold italic indicates the second-best performance, and the improvements represent relative improvements of using \modelname based on standard Transformer.

Model Num of Params SST-2 (test)IMDB Sentiment140 Amazon Reviews
Loss \downarrow Acc \uparrow Loss \downarrow Acc \uparrow Loss \downarrow Acc \uparrow Loss \downarrow Acc \uparrow
LSTM 120.14M 0.4760 80.60 0.6449 64.38 0.8026 59.79 0.5791 71.52
Mamba 129.73M 0.4335 79.59 0.6863 62.03 0.7871 58.74 0.6163 67.19
Transformer 109.48M 0.4297 81.19 0.5649 69.94 0.8891 57.79 0.5563 71.55
\hdashline w/ \modelname(Gated)95.33M 0.4250 80.39 0.5817 70.12 0.7941 61.94 0.4835 76.89
w/ \modelname 95.32M 0.4094 81.54 0.5225 73.98 0.8257 60.93 0.4748 77.63
Improvements\downarrow 14.16M\downarrow 4.72%\uparrow 0.43%\downarrow 7.51%\uparrow 5.78%\downarrow 7.13%\uparrow 5.43%\downarrow 14.65%\uparrow 8.50%

Results. We report the performance comparison between different sequence models across four sentiment analysis datasets, as shown in Table [3](https://arxiv.org/html/2410.02675v4#S4.T3 "Table 3 ‣ 4.2 Application of Real-world Task ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"). The results indicate that Transformer with \modelname achieves clear improvements compared to the standard Transformer and other baselines, such as LSTM and Mamba, especially for zero-shot OOD performance on IMDB, Sentiment140, and Amazon Reviewers datasets. Using \modelname achieves the relative improvements up to 14.65% and 8.50% in terms of Loss and Accuracy respectively, while reducing parameter numbers by about 14.16M. It indicates the potential of periodicity modeling to enhance both effectiveness and generalization on cross-domain language modeling and sentiment analysis tasks.

4) Image Recognition is a key computer vision task where image content is identified and categorized. Our evaluation contains four public benchmarks of image recognition: MNIST (LeCun et al., [2010](https://arxiv.org/html/2410.02675v4#bib.bib18)), MNIST-M(Ganin et al., [2016](https://arxiv.org/html/2410.02675v4#bib.bib6)), Fashion-MNIST(Xiao et al., [2017](https://arxiv.org/html/2410.02675v4#bib.bib51)), and Fashion-MNIST-C(Weiss & Tonella, [2022](https://arxiv.org/html/2410.02675v4#bib.bib49)), where MNIST-M and Fashion-MNIST-C are the variants for robustness.

Table 4: Results on image recognition tasks, where OOD Accuracy means the performance on other paired datasets and the Bold values indicate the highest values under the same metrics.

Dataset Accuracy \uparrow OOD Accuracy \uparrow
CNN w/ FAN CNN w/ FAN
MNIST 99.63 99.67 28.85 30.3
\hdashline MNIST-M 94.52 94.23 82.85 83.55
Fashion-MNIST 94.15 94.47 49.82 51.88
\hdashline Fashion-MNIST-C 88.61 88.82 91.45 91.59

Results. We apply FAN to image recognition tasks on four classic benchmarks, as shown in Table [4](https://arxiv.org/html/2410.02675v4#S4.T4 "Table 4 ‣ 4.2 Application of Real-world Task ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"). The results show that using FAN outperforms the standard CNN in OOD scenarios and most cases of in-domain (ID) scenarios. We believe that there are also some latent periodic features in image recognition tasks, and FAN’s ability to model these periodic features can help CNN achieve competitive or superior performance, especially in OOD scenarios.

### 4.3 Further Analysis of FAN

Comparison with Fourier-based Networks. We mainly compare with 1) Fourier Neural Network (FNN)(Silvescu, [1999](https://arxiv.org/html/2410.02675v4#bib.bib38)) using the cosine or sine function or their linear combinations as the activation function. 2) Fourier Series Neural Network (FSNN) is defined as Eq. ([3](https://arxiv.org/html/2410.02675v4#S3.E3 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks")), which shares the parameters and computation of sine and cosine part. 3) Fourier Transform Neural Network (FTNN) is a type of neural network that employs Fourier Transform to process the intermediate output in the neural network, such as FNO (Li et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib21)). We compare \modelname with these Fourier-based Networks in terms of their periodicity modeling abilities and general-purpose capabilities for language modeling.

![Image 7: Refer to caption](https://arxiv.org/html/2410.02675v4/x7.png)

Figure 6: Comparison \modelname with Fourier-based Networks on complex periodicity modeling (y=e^{\sin(\pi x)^{2}+\cos(x)+(x\mod 3)-1}) and language modeling tasks.

As shown in Figure [6](https://arxiv.org/html/2410.02675v4#S4.F6 "Figure 6 ‣ 4.3 Further Analysis of FAN ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"), only FAN achieves excellent performance on both tasks, indicating the superiority of our specially designed architecture of FAN. In contrast, FNN and FSNN cannot fit language modeling tasks, which aligns with previous work (Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib47); Liu et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib25)) and our findings derived from Eq. ([3](https://arxiv.org/html/2410.02675v4#S3.E3 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks"))-([5](https://arxiv.org/html/2410.02675v4#S3.E5 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks")). Moreover, FTNN performs poorly on complex periodic modeling tasks, akin to MLP. This may be attributed to the fact that FTNN does not incorporate the Fourier principle into the network but applies Fourier Transform as an intermediate processing step, which disadvantages FTNN in capturing periodicity. From Table [5](https://arxiv.org/html/2410.02675v4#S4.T5 "Table 5 ‣ 4.3 Further Analysis of FAN ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"), FAN also achieves fewer parameters and better performance than FTNN in language modeling tasks.

Table 5: Comparison \modelname with Fourier-based Networks on language modeling tasks, where each of them replaces the MLP layer in the standard transformer and ID means in-domain.

Model Num of Params Loss \downarrow
Train ID Test OOD Test
MLP 109.48M 0.2574 0.4297 0.5649
FNN 109.48M 0.6933 0.7103 0.7135
FSNN 95.32M 0.6931 0.7210 0.7249
FTNN 300.56M 0.2449 0.4547 0.8128
\hdashline FAN 95.32M 0.2434 0.4094 0.5225

#### Runtime of FAN.

We analyze the actual running time of FAN layer compared to MLP Layer with different input and output dimensions, as shown in Table [6](https://arxiv.org/html/2410.02675v4#S4.T6 "Table 6 ‣ Runtime of FAN. ‣ 4.3 Further Analysis of FAN ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"). The experimental results show that MLPs exhibit smaller runtimes when the input and output sizes are small, due to PyTorch’s optimization of MLP. However, as the input and output sizes continue to increase, matrix computations become the main contributor to runtime. At this point, FAN’s fewer parameters and reduced FLOPs begin to show significant advantages. Note that FAN can be further optimized from the underlying implementation.

Table 6: Comparison of actual runtime between FAN and MLP.

1024\times 1024 2048\times 2048 4096\times 4096 8192\times 8192
MLP 0.064 ms 0.114 ms 0.212 ms 0.938 ms
FAN 0.128 ms 0.133 ms 0.211 ms 0.704 ms

#### The impact of hyperparameter \mathbf{d_{\text{p}}}.

In our experiments, we fix d_{\text{p}}=\frac{1}{4}d_{h} intuitively for FAN, where d_{h} denotes the dimension of hidden layers. As shown in Figure [8](https://arxiv.org/html/2410.02675v4#A2.F8 "Figure 8 ‣ B.4 The influence of hyperparameters 𝐝_\"p\" ‣ Appendix B Additional Experiments ‣ \modelname: Fourier Analysis Networks") of Appendix, we investigate the impact of varying d_{\text{p}} empirically on task performance by changing itself. The results indicate that performance initially improves as d_{\text{p}} increases, but then decreases beyond a certain point. This trend may be attributed to the number of potential periodic features specific to each task. Furthermore, there remains room for further improvements with the better setup of \mathbf{d_{\text{p}}}.

## 5 Related Work

In this section, we outline the two most relevant directions and associated papers of this work.

#### Learning Periodicity with Neural Networks.

Periodic functions are one of the most basic functions of importance to human society and natural science(Newton, [1687](https://arxiv.org/html/2410.02675v4#bib.bib29); Osborn & Sensier, [2002](https://arxiv.org/html/2410.02675v4#bib.bib33); Kwasnicki, [2008](https://arxiv.org/html/2410.02675v4#bib.bib15); De Groot & Franses, [2012](https://arxiv.org/html/2410.02675v4#bib.bib3); Zhang et al., [2017](https://arxiv.org/html/2410.02675v4#bib.bib52)). However, commonly used neural networks, such as MLPs and transformers, struggle with modeling periodicity. This limitation is attributed to the lack of inherent “periodicity” in their inductive biases. Some previous works(Silvescu, [1999](https://arxiv.org/html/2410.02675v4#bib.bib38); Liu, [2013](https://arxiv.org/html/2410.02675v4#bib.bib23); Parascandolo et al., [2016](https://arxiv.org/html/2410.02675v4#bib.bib34); Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib47)) proposed merely using standard periodic functions themselves or their linear combinations as activation functions, which only work well on some shallow and simple models. On this basis, work (Liu et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib25)) introduced the Snake function, i.e., x+\sin^{2}(x), as the activation function. However, we observed that it can fit periodic functions to a certain extent, but its effect is limited especially for OOD scenarios, as demonstrated in Appendix [D](https://arxiv.org/html/2410.02675v4#A4 "Appendix D Comparison of \modelname and Snake Activation Function ‣ \modelname: Fourier Analysis Networks"). Therefore, although some previous studies have attempted to integrate periodic information into neural networks, their actual performance and range of applications remain heavily constrained.

#### Fourier-based Neural Network.

Previous studies have explored Fourier-based neural networks, but these networks generally perform well on specific tasks, while their performance on more general tasks tends to be poorer (Zuo & Cai, [2005](https://arxiv.org/html/2410.02675v4#bib.bib55); Tan, [2006](https://arxiv.org/html/2410.02675v4#bib.bib42); Uteuliyeva et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib47); Jiang et al., [2022](https://arxiv.org/html/2410.02675v4#bib.bib14); Chen et al., [2022](https://arxiv.org/html/2410.02675v4#bib.bib2)). Fourier Neural Networks employ the cosine (Silvescu, [1999](https://arxiv.org/html/2410.02675v4#bib.bib38); Ngom & Marin, [2021](https://arxiv.org/html/2410.02675v4#bib.bib30)) or sin function (Parascandolo et al., [2016](https://arxiv.org/html/2410.02675v4#bib.bib34); Sitzmann et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib39)) or their combination (Liu, [2013](https://arxiv.org/html/2410.02675v4#bib.bib23)) as the activation function. Some work employs Fourier Transform to process the intermediate output of network (Li et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib21); Lee-Thorp et al., [2022](https://arxiv.org/html/2410.02675v4#bib.bib20)), but they did not address the challenges of periodicity modeling as verified in Figure [6](https://arxiv.org/html/2410.02675v4#S4.F6 "Figure 6 ‣ 4.3 Further Analysis of FAN ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"). Some researches focus on leveraging the network to simulate the formula of Fourier Series (Rafajłowicz & Pawlak, [1997](https://arxiv.org/html/2410.02675v4#bib.bib35); Halawa, [2008](https://arxiv.org/html/2410.02675v4#bib.bib8); Lee et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib19)), which generally possess a similar principle as Eq. ([3](https://arxiv.org/html/2410.02675v4#S3.E3 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks")). However, this leads to the same problem as in Eq. ([5](https://arxiv.org/html/2410.02675v4#S3.E5 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks")), i.e., they are hard to serve as building blocks for deep neural networks, which limits these approaches’ capabilities.

In this paper, we design \modelname to address these challenges, which performs exceptionally well on periodicity modeling and maintains broad applicability on real-world tasks.

## 6 Discussion

In this section, we have a broad discussion on the expressive power, extrapolation capability, and application scope of \modelname as follows.

First, \modelname theoretically possesses the equal expressive power as MLP since it also adheres to Universal Approximation Theorem, which guarantees its capacity for functional approximation (refer to Appendix [E](https://arxiv.org/html/2410.02675v4#A5 "Appendix E How FAN Comply with Universal Approximation Theorem ‣ \modelname: Fourier Analysis Networks") for the detailed explanation). Moreover, \modelname introduces an important enhancement by incorporating periodicity, a feature absent in MLPs. By leveraging this special design, \modelname not only retains the capabilities of MLP but also enhances its ability to capture periodic characteristics in data.

Second, we observe that existing networks often exhibit divergent predictions in OOD scenarios, as shown in Figure [3](https://arxiv.org/html/2410.02675v4#S4.F3 "Figure 3 ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"), Figure [4](https://arxiv.org/html/2410.02675v4#S4.F4 "Figure 4 ‣ 4 Experiments ‣ \modelname: Fourier Analysis Networks"), and Figure [7](https://arxiv.org/html/2410.02675v4#A1.F7 "Figure 7 ‣ Appendix A MLP ‣ \modelname: Fourier Analysis Networks") for periodicity modeling tasks. In contrast, \modelname demonstrates strong OOD extrapolation ability in both periodicity modeling and some real-world tasks. This extrapolation ability indicates that the network is no longer restricted to the paradigms present in training dataset, but instead exhibits a kind of “transboundary thinking”. This could be an important avenue for improving generalization and learning efficiency.

Third, beyond tasks that explicitly require periodicity modeling, \modelname also has utility in a broader range of applications, which has been evidenced by our extensive experiments on real-world tasks, such as symbolic formula representation, time series forecasting, language modeling, and image recognition, where \modelname achieve competitive or superior performance than Transformers and other baselines. In fact, many machine learning tasks may harbor hidden forms of periodicity, even without explicitly including periodicity, such as mathematical operations and logic reasoning. If the neural network lacks the ability to model periodicity, it could impair its learning efficiency, as demonstrated in Figure LABEL:SL. From a deeper perspective, periodicity is not just a data feature but reflects a form of structural knowledge — one that allows for the transfer and reuse of abstract rules and principles across different contexts.

## 7 Conclusion

In this paper, we have proposed Fourier Analysis Network (\modelname), a novel network that addresses periodicity modeling in existing networks while maintaining the general-purpose modeling capability. Experimental results demonstrate that \modelname successfully fit both basic and complex periodic functions, whereas other general-purpose networks failed. Moreover, using \modelname exhibit clear improvements in real-world tasks, such as symbolic formula representation, time series forecasting, language modeling, and image recognition, outperforming neural networks such as MLP, Transformer, KAN, LSTM, Mamba, and CNN. These promising results, especially the stronger performance and the fewer parameters and FLOPs compared to MLP, suggest its potential to become a key component of foundational models.

In future work, we aim to expand the application scope of FAN and further explore its theoretical foundations for various tasks such as language modeling.

## 8 Acknowledgement

We would like to thank Lecheng Wang and Xuanming Zhang for their participation in discussions related to this work.

## References

*   Bassey et al. (2021) Joshua Bassey, Lijun Qian, and Xiangfang Li. A survey of complex-valued neural networks. _CoRR_, abs/2101.12249, 2021. 
*   Chen et al. (2022) Hanlong Chen, Luzhe Huang, Tairan Liu, and Aydogan Ozcan. Fourier imager network (FIN): A deep neural network for hologram reconstruction with superior external generalization. _Light: Science & Applications_, 2022. 
*   De Groot & Franses (2012) Bert De Groot and Philip Hans Franses. Common socio-economic cycle periods. _Technological Forecasting and Social Change_, 79(1):59–68, 2012. 
*   Devlin et al. (2018) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. _CoRR_, abs/1810.04805, 2018. URL [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805). 
*   Duoandikoetxea (2024) Javier Duoandikoetxea. _Fourier analysis_, volume 29. American Mathematical Society, 2024. 
*   Ganin et al. (2016) Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor S. Lempitsky. Domain-adversarial training of neural networks. _J. Mach. Learn. Res._, 17:59:1–59:35, 2016. 
*   Gu & Dao (2023) Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. _CoRR_, abs/2312.00752, 2023. 
*   Halawa (2008) Krzysztof Halawa. Fast and robust way of learning the fourier series neural networks on the basis of multidimensional discrete fourier transform. In _ICAISC_, volume 5097 of _Lecture Notes in Computer Science_, pp. 62–70. Springer, 2008. 
*   Han et al. (2022) Bing Han, Cheng Wang, and Kaushik Roy. Oscillatory fourier neural network: A compact and efficient architecture for sequential processing. In _AAAI_, pp. 6838–6846. AAAI Press, 2022. 
*   Haykin (1998) Simon Haykin. _Neural networks: a comprehensive foundation_. Prentice Hall PTR, 1998. 
*   Hendrycks & Gimpel (2016) Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). _arXiv preprint arXiv:1606.08415_, 2016. 
*   Hochreiter & Schmidhuber (1997) Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. _Neural Comput._, 9(8):1735–1780, 1997. 
*   Hornik et al. (1989) Kurt Hornik, Maxwell B. Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. _Neural Networks_, 2(5):359–366, 1989. 
*   Jiang et al. (2022) Song Jiang, Tahin Syed, Xuan Zhu, Joshua Levy, Boris Aronchik, and Yizhou Sun. Bridging self-attention and time series decomposition for periodic forecasting. In _CIKM_, pp. 3202–3211. ACM, 2022. 
*   Kwasnicki (2008) Witold Kwasnicki. Kitchin, juglar and kuznetz business cycles revisited. _Wroclaw: Institute of Economic Sciences_, 2008. 
*   Lai et al. (2018) Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-term temporal patterns with deep neural networks. In _SIGIR_, pp. 95–104. ACM, 2018. 
*   LeCun et al. (1998) Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. _Proc. IEEE_, 86(11):2278–2324, 1998. 
*   LeCun et al. (2010) Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. _ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist_, 2, 2010. 
*   Lee et al. (2021) Jiyoung Lee, Wonjae Kim, Daehoon Gwak, and Edward Choi. Conditional generation of periodic signals with fourier-based decoder. _CoRR_, abs/2110.12365, 2021. 
*   Lee-Thorp et al. (2022) James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontañón. Fnet: Mixing tokens with fourier transforms. In _NAACL-HLT_, pp. 4296–4313. Association for Computational Linguistics, 2022. 
*   Li et al. (2021) Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In _ICLR_. OpenReview.net, 2021. 
*   Linden et al. (2003) Greg Linden, Brent Smith, and Jeremy York. Amazon.com recommendations: Item-to-item collaborative filtering. _IEEE Internet Comput._, 7(1):76–80, 2003. 
*   Liu (2013) Shuang Liu. Fourier neural network for machine learning. In _ICMLC_, pp. 285–290. IEEE, 2013. 
*   Liu et al. (2024) Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Y. Hou, and Max Tegmark. KAN: kolmogorov-arnold networks. _CoRR_, abs/2404.19756, 2024. 
*   Liu et al. (2020) Ziyin Liu, Tilman Hartwig, and Masahito Ueda. Neural networks fail to learn periodic functions and how to fix it. In _NeurIPS_, 2020. 
*   Loshchilov & Hutter (2019) Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR (Poster)_. OpenReview.net, 2019. 
*   Maas et al. (2011) Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In _ACL_, pp. 142–150. The Association for Computer Linguistics, 2011. 
*   Mildenhall et al. (2020) Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In _ECCV (1)_, volume 12346 of _Lecture Notes in Computer Science_, pp. 405–421. Springer, 2020. 
*   Newton (1687) Isaac Newton. _Philosophiae naturalis principia mathematica_. William Dawson & Sons Ltd., London, 1687. 
*   Ngom & Marin (2021) Marieme Ngom and Oana Marin. Fourier neural networks as function approximators and differential equation solvers. _Stat. Anal. Data Min._, 14(6):647–661, 2021. 
*   OpenAI (2023) OpenAI. GPT-4 technical report. _CoRR_, abs/2303.08774, 2023. 
*   Oreshkin et al. (2020) Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: neural basis expansion analysis for interpretable time series forecasting. In _ICLR_. OpenReview.net, 2020. 
*   Osborn & Sensier (2002) Denise R. Osborn and Marianne Sensier. The prediction of business cycle phases: Financial variables and international linkages. _National Institute Economic Review_, 182(1):96–105, 2002. doi: 10.1177/002795010218200110. URL [https://doi.org/10.1177/002795010218200110](https://doi.org/10.1177/002795010218200110). 
*   Parascandolo et al. (2016) Giambattista Parascandolo, Heikki Huttunen, and Tuomas Virtanen. Taming the waves: sine as activation function in deep neural networks. 2016. 
*   Rafajłowicz & Pawlak (1997) E Rafajłowicz and M Pawlak. On function recovery by neural networks based on orthogonal expansions. _Nonlinear Analysis: Theory, Methods & Applications_, 30(3):1343–1354, 1997. 
*   Rosenblatt (1958) Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain. _Psychological review_, 65(6):386, 1958. 
*   Sahni et al. (2017) Tapan Sahni, Chinmay Chandak, Naveen Reddy Chedeti, and Manish Singh. Efficient twitter sentiment classification using subjective distant supervision. In _COMSNETS_, pp. 548–553. IEEE, 2017. 
*   Silvescu (1999) Adrian Silvescu. Fourier neural networks. In _IJCNN_, pp. 488–491. IEEE, 1999. 
*   Sitzmann et al. (2020) Vincent Sitzmann, Julien N.P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In _NeurIPS_, 2020. 
*   Socher et al. (2013) Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In _EMNLP_, pp. 1631–1642. ACL, 2013. 
*   Stein & Weiss (1971) Elias M Stein and Guido Weiss. _Introduction to Fourier analysis on Euclidean spaces_, volume 1. Princeton university press, 1971. 
*   Tan (2006) HS Tan. Fourier neural networks and generalized single hidden layer networks in aircraft engine fault diagnostics. 2006. 
*   Tancik et al. (2020) Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In _NeurIPS_, 2020. 
*   Tolstov (2012) Georgi P Tolstov. _Fourier series_. Courier Corporation, 2012. 
*   Touvron et al. (2023) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. _CoRR_, abs/2307.09288, 2023. 
*   Ulyanov et al. (2016) Dmitry Ulyanov, Andrea Vedaldi, and Victor S. Lempitsky. Instance normalization: The missing ingredient for fast stylization. _CoRR_, abs/1607.08022, 2016. 
*   Uteuliyeva et al. (2020) Malika Uteuliyeva, Abylay Zhumekenov, Rustem Takhanov, Zhenisbek Assylbekov, Alejandro J. Castro, and Olzhas Kabdolov. Fourier neural networks: A comparative study. _Intell. Data Anal._, 24(5):1107–1120, 2020. 
*   Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NIPS_, pp. 5998–6008, 2017. 
*   Weiss & Tonella (2022) Michael Weiss and Paolo Tonella. Simple techniques work surprisingly well for neural network test prioritization and active learning. In _Proceedings of the 31th ACM SIGSOFT International Symposium on Software Testing and Analysis_, 2022. 
*   Wu et al. (2021) Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. _Advances in neural information processing systems_, 34:22419–22430, 2021. 
*   Xiao et al. (2017) Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. _CoRR_, abs/1708.07747, 2017. URL [http://arxiv.org/abs/1708.07747](http://arxiv.org/abs/1708.07747). 
*   Zhang et al. (2017) Liheng Zhang, Charu Aggarwal, and Guo-Jun Qi. Stock price prediction via discovering multi-frequency trading patterns. In _Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, KDD ’17, pp. 2141–2149, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450348874. doi: 10.1145/3097983.3098117. URL [https://doi.org/10.1145/3097983.3098117](https://doi.org/10.1145/3097983.3098117). 
*   Zhou et al. (2021) Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In _AAAI_, pp. 11106–11115. AAAI Press, 2021. 
*   Zhou et al. (2022) Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In _ICML_, volume 162 of _Proceedings of Machine Learning Research_, pp. 27268–27286. PMLR, 2022. 
*   Zuo & Cai (2005) Wei Zuo and Lilong Cai. Tracking control of nonlinear systems using fourier neural network. In _Proceedings, 2005 IEEE/ASME International Conference on Advanced Intelligent Mechatronics._, pp. 670–675. IEEE, 2005. 

## Appendix A MLP

The MLP layer \Phi(x) is defined as:

\Phi(x)=\sigma(B_{m}+W_{m}x),(12)

where B_{m}\in\mathbb{R}^{d_{m}} and W_{\bar{p}}\in\mathbb{R}^{d_{x}\times d_{m}} are learnable parameters with the hyperparameter d_{m} indicating the first dimension of W_{m}, \sigma denotes the activation function, and MLP can be defined as the stacking of the MLP layer \Phi(x):

\text{MLP}(x)=\Phi_{L}\circ\Phi_{L-1}\circ\cdots\circ\Phi_{1}\circ x,(13)

where

\Phi_{l}(x)=\left\{\begin{array}[]{ll}\sigma(B_{m}^{l}+W^{l}_{m}x),&\text{if }%
l<L,\\
B^{L}+W^{L}x,&\text{if }l=L.\end{array}\right.(14)

![Image 8: Refer to caption](https://arxiv.org/html/2410.02675v4/x8.png)

![Image 9: Refer to caption](https://arxiv.org/html/2410.02675v4/x9.png)

![Image 10: Refer to caption](https://arxiv.org/html/2410.02675v4/x10.png)

Figure 7: The performance of \modelname in periodicity modeling compared to MLP, KAN, and Transformer (Part II), where the green line represents the test data within the domain of training data, while the blue line represents the test data outside the domain of training data.

Table 7: Performance of different sequence models on time series forecasting tasks, where Input Length = 96, the bold values indicate the lowest value on each row, and Improve means the relative improvements of using \modelname and \modelname(Gated) based on standard Transformer.

Dataset Output Length LSTM(12.51 M)Mamba(12.69 M)Transformer(12.12 M)Transformer with \modelname(11.06 M)
Gated Default
MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow
Weather 96 1.069 0.742 0.552 0.519 0.413 0.438 0.292 0.380 0.313 0.431
192 1.090 0.778 0.700 0.595 0.582 0.540 0.535 0.550 0.472 0.525
336 0.992 0.727 0.841 0.667 0.751 0.626 0.637 0.602 0.719 0.581
720 1.391 0.892 1.171 0.803 0.967 0.715 0.845 0.706 0.732 0.670
Exchange 96 0.938 0.794 0.908 0.748 0.777 0.681 0.685 0.644 0.657 0.623
192 1.241 0.899 1.328 0.925 1.099 0.800 0.998 0.757 0.968 0.741
336 1.645 1.048 1.512 0.992 1.614 1.029 1.511 0.961 1.266 0.905
720 1.949 1.170 2.350 1.271 2.163 1.204 1.658 1.104 1.857 1.145
Traffic 96 0.659 0.359 0.666 0.377 0.656 0.357 0.647 0.355 0.643 0.347
192 0.668 0.360 0.671 0.381 0.672 0.363 0.649 0.353 0.657 0.354
336 0.644 0.342 0.665 0.374 0.673 0.360 0.665 0.358 0.656 0.353
720 0.654 0.351 0.662 0.364 0.701 0.380 0.682 0.369 0.673 0.363
ETTh 96 0.999 0.738 0.860 0.697 1.139 0.853 0.842 0.736 0.873 0.707
192 1.059 0.759 0.849 0.700 1.373 0.932 0.885 0.748 0.914 0.741
336 1.147 0.820 1.005 0.745 1.261 0.924 0.980 0.770 0.999 0.793
720 1.206 0.847 0.994 0.758 1.056 0.819 1.002 0.798 1.031 0.818
Average(Improve)–1.083 0.726 1.002 0.668 0.994 0.689 0.845\downarrow 15.0%0.637\downarrow 7.6%0.839\downarrow 15.6%0.631\downarrow 8.4%

## Appendix B Additional Experiments

### B.1 Additional Experiments on Periodicity Modeling Tasks.

More experimental results on periodicity modeling tasks are shown in Figure [7](https://arxiv.org/html/2410.02675v4#A1.F7 "Figure 7 ‣ Appendix A MLP ‣ \modelname: Fourier Analysis Networks").

### B.2 FAN for Solving SciML Problems

We conduct experiments on the SciML problem that includes the Fourier function class following the work (Li et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib21)). The Burgers’ equation, a non-linear partial differential equation, is frequently used in scientific computing to model shock waves and traffic flow, among other phenomena. The detailed error rate on Burgers’ equation is listed in the Table [8](https://arxiv.org/html/2410.02675v4#A2.T8 "Table 8 ‣ B.2 FAN for Solving SciML Problems ‣ Appendix B Additional Experiments ‣ \modelname: Fourier Analysis Networks"). We can find that replacing the MLP Layer with FAN Layer in Fourier Neural Operator (FNO) (Li et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib21)) can achieve clear improvements on each setting of resolution s of this task.

Table 8: The error rate on Burgers’ equation. The values in the table represent the Average Relative Error for Burgers’ equation with lower values indicating better performance.

Model s=256 s=512 s=1024 s=2048 s=4096 s=8192
FNO 5.93%6.14%6.03%6.75%7.36%9.93%
FNO with FAN 5.26%5.17%5.18%6.73%6.35%7.06%

### B.3 Comparison with Directly Learning the Coefficients

We compare FAN with a baseline of directly learning the coefficients, which inputs sin(x) and cos(x) and then uses the MLP Layer instead of the FAN Layer to model the Fourier coefficients. In this setting, frequencies are fixed and only the coefficients are learned, which may limit the model’s ability to capture patterns not aligned with these frequencies. Taking simple f(x)=x\ mod\ 5 as an example, this setting may not even converge at all, because the frequency of x\ mod\ 5 is inconsistent with sin(x) and cos(x). The experimental results of their loss are shown in Table [9](https://arxiv.org/html/2410.02675v4#A2.T9 "Table 9 ‣ B.3 Comparison with Directly Learning the Coefficients ‣ Appendix B Additional Experiments ‣ \modelname: Fourier Analysis Networks").

Table 9: Comparison of FAN and directly learning the coefficients on fitting f(x)=x\ mod\ 5.

Epoch 50 100 150 200
Directly learning the coefficients 2.10 2.09 2.09 2.08
FAN 0.28 0.23 0.18 0.17

### B.4 The influence of hyperparameters \mathbf{d_{\text{p}}}

We evaluate the influence of hyperparameters \mathbf{d_{\text{p}}} as shown in Figure [8](https://arxiv.org/html/2410.02675v4#A2.F8 "Figure 8 ‣ B.4 The influence of hyperparameters 𝐝_\"p\" ‣ Appendix B Additional Experiments ‣ \modelname: Fourier Analysis Networks").

![Image 11: Refer to caption](https://arxiv.org/html/2410.02675v4/x11.png)

Figure 8: The influence of hyper-parameters \mathbf{d_{\text{p}}} on language modeling tasks. We use the red dashed line to represent the performance of the standard Transformer.

### B.5 The effectiveness of the FAN Layer for deep neural networks

We evaluate the effect of varying the number of FAN layers from 3 to 20 on periodicity modeling tasks, employing residual connections to mitigate overfitting. The experimental results show that both the best training loss and test loss still decrease slowly as the number of layers increases.

Furthermore, on Language Modeling tasks, we replaced 24 MLP Layers of Transformer with 24 FAN Layers, i.e. Transformer with FAN, and it also achieved clear improvements on each task, especially for OOD zero-shot evaluation scenarios. These findings indicate that FAN Layer is effective for deep neural networks.

![Image 12: Refer to caption](https://arxiv.org/html/2410.02675v4/x12.png)

Figure 9: Performance of Deeper FAN on fitting y=e^{\sin^{2}(\pi x)+\cos(x)+(x\mod 3)}-1.

### B.6 Comparison with Frequency-based Models in Time Series Forecasting Tasks

To compare with frequency-based models in Time Series Forecasting tasks such as FEDformer (Zhou et al., [2022](https://arxiv.org/html/2410.02675v4#bib.bib54)), we replace MLP with FAN in frequency-based models. We present the experimental results in Table [10](https://arxiv.org/html/2410.02675v4#A2.T10 "Table 10 ‣ B.6 Comparison with Frequency-based Models in Time Series Forecasting Tasks ‣ Appendix B Additional Experiments ‣ \modelname: Fourier Analysis Networks"), where the results of FEDformer are cited from its paper directly. From the results, we can find that FEDformer with FAN can outperform FEDformer in almost all cases.

Table 10: Results of comparison with frequency-based models in time series forecasting tasks.

Dataset Len FEDformer with FAN
MSE MAE MSE MAE
Traffic 96 0.587 0.366 0.577 0.357
192 0.604 0.373 0.601 0.366
336 0.621 0.383 0.620 0.378
720 0.626 0.382 0.619 0.370
Exchange 96 0.148 0.278 0.138 0.267
192 0.271 0.380 0.261 0.371
336 0.460 0.500 0.461 0.503
720 1.195 0.841 1.159 0.827
Electricity 96 0.193 0.308 0.184 0.298
192 0.201 0.315 0.199 0.313
336 0.214 0.329 0.212 0.325
720 0.246 0.355 0.239 0.347

### B.7 Experiments on Time Series Forecasting with Instance Normalization

We conduct experiments on time series forecasting tasks with instance normalization (Ulyanov et al., [2016](https://arxiv.org/html/2410.02675v4#bib.bib46)), and the results are shown in Table [11](https://arxiv.org/html/2410.02675v4#A2.T11 "Table 11 ‣ B.7 Experiments on Time Series Forecasting with Instance Normalization ‣ Appendix B Additional Experiments ‣ \modelname: Fourier Analysis Networks"). We find that applying instance normalization before the architecture can effectively improve the performance.

Table 11: Results on time series forecasting tasks with instance normalization, where Input Length = 96, the bold values indicate the lowest value on each row, and the improve means the relative improvements of using \modelname and \modelname(Gated) based on Transformer.

Dataset Output Length Transformer(12.12 M)Transformer with \modelname(11.06 M)
Gated Default
MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow MSE \downarrow MAE \downarrow
Weather 96 0.1772 0.2301 0.1864 0.2352 0.1756 0.2247
192 0.2438 0.2844 0.2445 0.2834 0.2327 0.2760
336 0.3077 0.3267 0.3156 0.3320 0.3118 0.3291
720 0.4253 0.3982 0.3909 0.3782 0.4113 0.3906
Exchange 96 0.1433 0.2653 0.1157 0.2452 0.1436 0.2666
192 0.2563 0.3552 0.2539 0.3611 0.2651 0.3757
336 0.5273 0.5218 0.4329 0.4891 0.5092 0.5326
720 1.7401 0.9273 1.5783 0.9303 1.0599 0.7657
Traffic 96 0.6160 0.3449 0.6030 0.3334 0.6109 0.3319
192 0.6329 0.3479 0.6239 0.3404 0.6258 0.3370
336 0.6369 0.3485 0.6416 0.3487 0.6200 0.3380
720 0.6555 0.3577 0.6645 0.3574 0.6412 0.3525
ETTh1 96 0.5339 0.4910 0.5503 0.5216 0.5378 0.4983
192 0.5633 0.5209 0.5906 0.5346 0.5968 0.5265
336 0.7576 0.5813 0.6640 0.5636 0.7525 0.5933
720 0.7411 0.6177 0.7411 0.6066 0.7328 0.6142
ETTh2 96 0.3881 0.4097 0.4082 0.4292 0.3833 0.4149
192 0.5766 0.4999 0.4695 0.4514 0.5039 0.4640
336 0.5782 0.5100 0.5556 0.5012 0.5417 0.4940
720 0.5841 0.5230 0.5070 0.4943 0.5272 0.4951
Average(Improve)–0.554 0.444 0.526\downarrow 5.1%0.436\downarrow 1.9%0.509\downarrow 8.2%0.430\downarrow 3.2%

## Appendix C Experimental Details

### C.1 Implementation Details.

We conduct our experiments on eight GPU of Tesla A100-PCIe-40G. Unless otherwise specified, we use the following hyperparameters in the experiments. The model architecture consists of 3 to 24 layers, the activation function \sigma is set to GELU (Hendrycks & Gimpel, [2016](https://arxiv.org/html/2410.02675v4#bib.bib11)), and the dimension of the projection matrix W_{p} is set to d_{p}=\frac{1}{4}d_{h}, where d_{h} denotes the dimension of the hidden layers. We employ the AdamW optimizer (Loshchilov & Hutter, [2019](https://arxiv.org/html/2410.02675v4#bib.bib26)) for the model’s training process.

### C.2 Setup of Periodicity Modeling

In periodicity modeling tasks, \modelname, MLP, and KAN each consist of three layers with comparable FLOPs, while the Transformer model comprises twelve layers. For consistency, we set the hidden layer dimension (d_{h}) to 2048 for \modelname, MLP, and Transformer. In the case of KAN, we follow its original paper (Liu et al., [2024](https://arxiv.org/html/2410.02675v4#bib.bib24)), where the spline order (K) and the number of spline intervals (G) are set to 3 and 50, respectively. We apply a learning rate of 1\times 10^{-5} for training all models. We ensured that the data density of each period in tasks was consistent, meaning that each cycle contained a fixed quantity of 10,000 training data points.

### C.3 Setup of Symbolic Formula Representation

In symbolic formula representation tasks, we used the create_dataset function from the official KAN repository to generate the datasets. Each dataset contains 3000 training samples and 1000 test samples, with all input variables randomly sampled from the range [-1, 1]. We followed the training settings from the original KAN paper, training all methods using LBFGS for 1800 steps. For KAN, we increased the number of grid points to scale up the parameter size, covering G=\{3,5,10,20,50,100,200,500,1000\}. For other methods, we scaled up the parameter size by increasing the number of layers and the dimensions of hidden layers.

### C.4 Setup of Time Series Forecasting

In time series forecasting task, we implement our model based on the codebase by (Wu et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib50)). Each model comprises 2 encoder layers and 1 decoder layer. We fix the hidden size for both the Transformer and our model at 512, with the feedforward dimension set to 2048 (four times the hidden size). The parameter sizes detailed in the main text correspond to the Exchange dataset; variations in the number of variables across different datasets influence the linear layers in the model. We adjust the hidden sizes of the other models to align with the Transformer parameters for fairness.

### C.5 Setup of Language Modeling

In language modeling task, we employ the BERT tokenizer (Devlin et al., [2018](https://arxiv.org/html/2410.02675v4#bib.bib4)) and an embedding layer with a dimensionality of 768, except for Mamba, which adheres to its default settings as specified in the original paper (Gu & Dao, [2023](https://arxiv.org/html/2410.02675v4#bib.bib7)). The architecture features 4, 24, and 12 layers with hidden sizes of 1800, 768, and 768 for LSTM, Mamba, and Transformers, respectively. To mitigate training stagnation in deeper LSTM models, we reduce the number of layers while increasing the hidden size to balance the parameters. Importantly, Mamba’s layer count is twice that of a similarly sized Transformer, as each layer consists of two Mamba blocks (Multihead attention block + MLP block).

### C.6 Setup of Image Recognition

In image recognition tasks, we employ a simple CNN as the baseline model, which consists of four Convolutional Layers and two MLP Layers. We replace MLP with FAN in CNN, i.e. CNN with FAN, as the counterpart, ensuring that they have similar parameters. For each task, we use stochastic gradient descent with momentum (SGDM) as the optimizer, the learning rate is set to 0.01, and the training process runs for 100 epochs.

## Appendix D Comparison of \modelname and Snake Activation Function

We compare FAN with Snake, a previous approach used for improving the fitting of periodic functions with neural networks. The results are shown in Figure [10](https://arxiv.org/html/2410.02675v4#A4.F10 "Figure 10 ‣ Appendix D Comparison of \modelname and Snake Activation Function ‣ \modelname: Fourier Analysis Networks").

![Image 13: Refer to caption](https://arxiv.org/html/2410.02675v4/x13.png)

![Image 14: Refer to caption](https://arxiv.org/html/2410.02675v4/x14.png)

Figure 10: Comparisons of \modelname with MLP (Snake) (Liu et al., [2020](https://arxiv.org/html/2410.02675v4#bib.bib25)) in fitting periodic functions.

## Appendix E How FAN Comply with Universal Approximation Theorem

The Universal Approximation Theorem states that a feed-forward network with a single hidden layer containing a (sufficiently large) finite number of neurons can approximate continuous functions on compact subsets of \mathbb{R}^{n}, under mild assumptions on the activation function, which needs to be a non-constant, no-linear, and continuous function. FAN Layer is defined as \phi(x)=[\cos(W_{p}x)||\sin(W_{p}x)||\sigma(B_{\bar{p}}+W_{\bar{p}}x)], where || denotes concatenation and \sigma denotes the standard activation function, such as ReLU and GELU. Since \sin and \cos functions also satisfy the required conditions of being non-constant, continuous, and non-linear activation functions, the FAN layer adheres to the Universal Approximation Theorem.

## Appendix F More Details of Baselines

In our experiments, we mainly compare \modelname with the following baselines. 1) MLP(Rosenblatt, [1958](https://arxiv.org/html/2410.02675v4#bib.bib36)): the most classic model, which is widely used in the backbone of various models. 2) Transformer(Vaswani et al., [2017](https://arxiv.org/html/2410.02675v4#bib.bib48)): a prevalent model known for its self-attention mechanism, which achieves outstanding performance on various tasks. 3) KAN(Liu et al., [2024](https://arxiv.org/html/2410.02675v4#bib.bib24)): an emerged model specialized for symbolic formula representation, which uses the b-spline functions instead of fixed activation functions. 4) LSTM(Hochreiter & Schmidhuber, [1997](https://arxiv.org/html/2410.02675v4#bib.bib12)): a well-known recurrent neural network (RNN) that can capture long-term dependencies on sequential data. 5) Mamba(Gu & Dao, [2023](https://arxiv.org/html/2410.02675v4#bib.bib7)): an emerged selective state space model (SSM) that achieves competitive performance on some tasks with sequential inputs. 6) CNN(LeCun et al., [1998](https://arxiv.org/html/2410.02675v4#bib.bib17)): convolutional neural network contains the convolutional layers, which are effective in processing image data.

For Fourier-based Networks, we mainly compare \modelname with 1) Fourier Neural Network (FNN) (Silvescu, [1999](https://arxiv.org/html/2410.02675v4#bib.bib38)) using the cosine or sine function or their linear combinations as the activation function. 2) Fourier Series Neural Network (FSNN) is defined as Eq. ([3](https://arxiv.org/html/2410.02675v4#S3.E3 "In 3 Fourier Analysis Network (\modelname) ‣ \modelname: Fourier Analysis Networks")), which shares the parameters and computation of Sin and Cos part. 3) Fourier Transform Neural Network (FTNN) is a type of neural network that employs Fourier Transform to process the intermediate output in the neural network, such as FNO (Li et al., [2021](https://arxiv.org/html/2410.02675v4#bib.bib21)).

[7] Title: Fourier Analysis Networks, Explained
[7] URL Source: https://medium.com/demistify/fourier-analysis-networks-explained-97b3fb2f16d3
[7] Description: Fourier Analysis Networks, Explained For most AI models today, their foundation is often built upon multi-layer perceptrons (MLPs), which is an artificial (feedforward) neural network with multiple …
[7] Published Time: 2025-01-29T20:19:20Z

[![Image 1: Sean Eugene Chua](https://miro.medium.com/v2/resize:fill:64:64/1*azJ3IbceJFhebT-3x_ovMA.png)](https://medium.com/@seanchua873?source=post_page---byline--97b3fb2f16d3---------------------------------------)

6 min read

Jan 29, 2025

--

For most AI models today, their foundation is often built upon multi-layer perceptrons (MLPs), which is an artificial (feedforward) neural network with multiple layers. These types of models often suffice for most types of supervised data and can achieve relatively high accuracy after setting the optimal parameters and hyperparameters. However, these models have a more difficult time predicting data that exhibits **periodicity**(i.e. the frequency of observations in data over time). MLPs treat inputs as independent and lack mechanisms to capture temporal or sequential patterns. MLPs struggle to effectively learn and predict periodic data since they do not possess the ability to model frequency, amplitude, or phase shifts in periodic signals. Fourier analysis networks (FANs) aim to resolve this current gap. Built on the mathematical principles of Fourier analysis, these networks show immense promise in handling structured and periodic data, particularly in applications like time-series forecasting and signal processing. This article goes into a conceptual exploration into FANs, highlighting their foundational concepts and applications in real-world scenarios.

## **Fourier Analysis: The Foundation**

Fourier analysis studies how general functions can be decomposed into trigonometric or exponential functions with defined frequencies. This decomposition is essential for understanding periodic and structured data, making Fourier analysis vital in scientific fields and engineering. For instance, periodic data such as audio signals, electromagnetic waves in communication systems, and repeating patterns in image processing (e.g., the compression of JPEG images) all rely on Fourier analysis in their applications. There are two main types of Fourier expansions:

1.   **Fourier Series:** Used for periodic functions, which can be expressed as a discrete sum of trigonometric or exponential terms with specific frequencies.
2.   **Fourier Transform:** Applicable to non-periodic functions, representing them as a continuous integral of trigonometric or exponential terms over a continuum of frequencies.

Press enter or click to view image in full size

Addition of Fourier Transforms (source: [3Blue1Brown](https://www.youtube.com/watch?v=spUNpyF58BY))

The power of Fourier analysis lies in its ability to simplify complex problems, especially on data containing some aspect of sinusoidal components or properties.

**Fourier Transforms and Periodic Data (such as Time Series Forecasting)**Fourier transforms are typically described as the exponential decomposition of a function. Its mathematical expression is found below. For a given function _f(t)_, the Fourier transform _F(ω)_ is defined as:

Press enter or click to view image in full size

Here, _f(t)_ represents the input function (for example, a signal), and ω is the angular frequency. Additionally, _F(ω)_ describes the signal in terms of its frequency components, showing which frequencies are present. From a signal processing perspective, given some signal, we can use the Fourier transform to deconstruct different parts of the signal into corresponding sine and cosine waves. With this, these are some key properties that the Fourier Transform has:

1.   **Frequency Decomposition:** It separates a signal into its constituent frequencies, helping identify “dominant” signal components.
2.   **Bidirectional Transformation:** We can obtain an inverse Fourier transform that allows reconstruction of the original signal given its frequency representation.

In the context of time-series forecasting, one of the most important techniques in predictive analytics within machine learning, Fourier Transforms are particularly valuable because they reveal the periodic components in a dataset, (i.e. time-based trends), enable signal filtering by Isolating specific frequencies to remove noise and extract relevant pattern, and are able to utilize such periodic components to accurately predict trends over time.

## Fourier Analysis Networks

Fourier analysis networks combine traditional neural layers with frequency-based representations, creating a robust hybrid model for structured data. At their core, they use sinusoidal functions to represent data in the form of frequencies. This integration is often achieved through **Fourier feature mappings**.

Fourier feature mapping is a technique that takes raw input data and transforms it into a higher-dimensional space using sine and cosine functions. This transformation allows the neural network to analyze the data in terms of its frequency components rather than just its raw values. Mathematically, according to Dong et al. (2024), the transformation can be expressed as:

Press enter or click to view image in full size

where Bp̄ and Wp̄ are learnable parameters; Bp̄ usually denotes a parameter that controls the frequencies of the sine and cosine functions used in the Fourier feature mapping, while Wp̄ is a vector that contains values that weight the influence of each Fourier basis function in the model.

## **How Are These Features Used in Neural Networks?**

Once the input data has been transformed using Fourier feature mapping, these new features are fed into a neural network. From this, the same principles apply with how neural networks learn via supervised learning. The model can recognize complex patterns more easily because it is working with features that highlight periodic behavior in time-series data (FANs decompose input data into multiple sinusoidal components). By using Fourier features, neural networks can converge faster during training and thus learn high-frequency signals more efficiently. More importantly, they are able to efficiently handle datasets with overlapping periodicities. Finally, by incorporating Fourier basis functions, the network generalizes better to unseen data.

## **Comparison with Traditional Approaches**

Traditional neural networks require more layers and parameters to approximate periodic patterns. Fourier networks achieve the same with fewer resources, as they explicitly model these patterns with sinusoidal functions. In effect, FANs help the neural network learn periodic or repetitive patterns more accurately and efficiently.

Press enter or click to view image in full size

Performance Comparison: MLP vs FAN (source: [Dong et al.](https://arxiv.org/abs/2410.02675))

Right off the bat, we can examine the equations and see that FANs are superior from an efficiency standpoint. Looking at the number of parameters in each layer, note that compared to the MLP layer, the FAN layer has an additional factor of 1-(d_p / d_output) which is less than 1. Consequently, the number of parameters in the FAN layer is less than that of the MLP layer. Similarly, looking at the number of floating-point operations (FLOPs) of both layers, the same can be said, with the FAN layer having an additional factor of 1-(d_p / d_output) which is less than 1.

Press enter or click to view image in full size

From a performance standpoint, we can see that FANs do a phenomenal job in modeling periodicity compared to other models like MLPs, Kolmogorov-Arnold networks (KANs) and Transformers. From Dong et al., the graph above shows that FANs do “exceptionally well on test data both within and outside the domain of the training data, indicating that it is genuinely modeling periodicity rather than merely memorizing the training data.”

## **Practical Applications of FANs**

FANs have proven especially effective in applications such as signal processing, stock market analysis, and seismology, where understanding frequency-based patterns is critical. In signal processing, FANs are used to filter noise, enhance audio quality, and compress data for efficient transmission. By analyzing signals and their frequencies, FANs can isolate relevant features that enable clearer communication or improved audio quality. In stock market analysis, FANs help detect cyclical trends and recurring patterns within financial time series data to more accurately predict market movements and identify potential opportunities based on frequency-driven price fluctuations. Finally, in seismology, FANs play a pivotal role in processing seismic data, separating low-frequency noise from high-frequency earthquake signals, which aids in real-time earthquake detection and prediction.

## **Final Thoughts**

Fourier networks and FANs represent a paradigm shift in how structured and periodic data is modeled in AI/ML. By embedding Fourier analysis into neural network architectures, they provide an efficient and accurate solution for predicting and analyzing periodic data in time-series forecasting and signal processing.

## **References**

3Blue1Brown. 2018. “But What Is the Fourier Transform? A Visual Introduction.” [https://www.youtube.com/watch?v=spUNpyF58BY](https://www.youtube.com/watch?v=spUNpyF58BY).

Dong, Yihong, Ge Li, Yongding Tao, Xue Jiang, Kechi Zhang, Jia Li, Jing Su, Jun Zhang, and Jingjing Xu. 2024. “FAN: Fourier Analysis Networks.” arXiv Preprint arXiv:2410.02675.

Hayes, Adam. 2025. “Fourier Analysis.” [https://www.investopedia.com/terms/f/fourieranalysis.asp](https://www.investopedia.com/terms/f/fourieranalysis.asp).

[8] Title: NeurIPS Poster FAN: Fourier Analysis Networks
[8] URL Source: https://neurips.cc/virtual/2025/poster/117474
[8] Description: 

[Skip to yearly menu bar](https://neurips.cc/virtual/2025/poster/117474#child-menu)[Skip to main content](https://neurips.cc/virtual/2025/poster/117474#main)

## Main Navigation

[![Image 1: conference_logo](https://neurips.cc/static/core/img/neurips-navbar-logo.svg)](https://neurips.cc/)

*   [NeurIPS](https://neurips.cc/virtual/2025/poster/117474#)
    *   [Help/FAQ](https://neurips.cc/FAQ)

* * *

    *   [Contact NeurIPS](https://neurips.cc/Help/Contact)

* * *

    *   [Create Profile](https://neurips.cc/Profile/create)

* * *

    *   [Code of Ethics](https://neurips.cc/Conferences/2023/EthicsGuidelines)

* * *

    *   [Code of Conduct](https://neurips.cc/public/CodeOfConduct)

* * *

    *   [Journal To Conference Track](https://neurips.cc/public/JournalToConference)

* * *

    *   [Diversity & Inclusion](https://neurips.cc/public/DiversityInclusion)

* * *

    *   [Proceedings](https://proceedings.neurips.cc/)

* * *

    *   [Future Meetings](https://neurips.cc/Conferences/FutureMeetings)

* * *

    *   [Press](https://neurips.cc/Conferences/2025/Press)

* * *

    *   [Exhibitor Information](https://neurips.cc/Exhibitors/exhibitorinfo)

* * *

    *   [Privacy Policy](https://neurips.cc/public/PrivacyPolicy)

* * *

    *   [Downloads](https://neurips.cc/Downloads)

*   [My Stuff](https://neurips.cc/MyStuff)

[Login](https://neurips.cc/accounts/login?nextp=/virtual/2025/loc/san-diego/poster/121402)

*   ![Image 2: San Diego graphic](https://neurips.cc/media/Locations/15-san-diego.svg) San Diego 
*   ![Image 3: Atlanta graphic](https://neurips.cc/media/)[Sydney](https://neurips.cc/virtual/2025/loc/sydney/poster/117474)
*   ![Image 4: Atlanta graphic](https://neurips.cc/media/)[Atlanta](https://neurips.cc/virtual/2025/loc/atlanta/poster/117474)
*   ![Image 5: Mexico City graphic](https://neurips.cc/media/Locations/17-mexico-city.svg)[Mexico City](https://neurips.cc/virtual/2025/loc/mexico-city/poster/117474)

*   [Select Year: (2025)](https://neurips.cc/virtual/2025/poster/117474#)
    *   [2026](https://neurips.cc/Conferences/2026)

* * *

    *   [2025](https://neurips.cc/Conferences/2025)

* * *

    *   [2024](https://neurips.cc/Conferences/2024)

* * *

    *   [2023](https://neurips.cc/Conferences/2023)

* * *

    *   [2022](https://neurips.cc/Conferences/2022)

* * *

    *   [2021](https://neurips.cc/Conferences/2021)

* * *

    *   [2020](https://neurips.cc/Conferences/2020)

* * *

    *   [2019](https://neurips.cc/Conferences/2019)

* * *

    *   [2018](https://neurips.cc/Conferences/2018)

* * *

    *   [2017](https://neurips.cc/Conferences/2017)

* * *

    *   [2016](https://neurips.cc/Conferences/2016)

* * *

    *   [2015](https://neurips.cc/Conferences/2015)

* * *

    *   [2014](https://neurips.cc/Conferences/2014)

* * *

    *   [2013](https://neurips.cc/Conferences/2013)

* * *

    *   [2012](https://neurips.cc/Conferences/2012)

* * *

    *   [2011](https://neurips.cc/Conferences/2011)

* * *

    *   [2010](https://neurips.cc/Conferences/2010)

* * *

    *   [2009](https://neurips.cc/Conferences/2009)

* * *

    *   [2008](https://neurips.cc/Conferences/2008)

* * *

    *   [2007](https://neurips.cc/Conferences/2007)

* * *

    *   [2006](https://neurips.cc/Conferences/2006)

* * *

    *   [Earlier Conferences](https://neurips.cc/Conferences/PastConferences)

*   [Start Here](https://neurips.cc/virtual/2025/loc/san-diego/index.html)
*   [Schedule](https://neurips.cc/virtual/2025/loc/san-diego/calendar)
*   [Tutorials](https://neurips.cc/virtual/2025/loc/san-diego/events/tutorial)
*   [Main Conference](https://neurips.cc/virtual/2025/poster/117474#)
    *   [Invited Talks](https://neurips.cc/virtual/2025/loc/san-diego/eventlistwithbios/Invited%20Talk)

* * *

    *   [Orals](https://neurips.cc/virtual/2025/loc/san-diego/events/oral)

* * *

    *   [Papers](https://neurips.cc/virtual/2025/papers.html)

* * *

    *   [Competitions](https://neurips.cc/virtual/2025/events/Competition)

* * *

    *   [Datasets & Benchmarks](https://neurips.cc/virtual/2025/loc/san-diego/events/datasets-benchmarks-2025)

* * *

    *   [Journal Track](https://neurips.cc/virtual/2025/loc/san-diego/events/2025-journal-track-papers)

* * *

    *   [Creative AI Track](https://neurips.cc/virtual/2025/events/creative-ai-2025)

* * *

    *   [Outstanding Paper Awards](https://neurips.cc/virtual/2025/awards_detail)

* * *

    *   [Creative AI](https://neurips.cc/virtual/2025/loc/san-diego/events/creative-ai-2025)

* * *

    *   [Spotlights](https://neurips.cc/virtual/2025/loc/san-diego/events/spotlights-2025)

* * *

    *   [Awards](https://neurips.cc/virtual/2025/loc/san-diego/awards_detail)

*   [Community](https://neurips.cc/virtual/2025/poster/117474#)
    *   [Affinity Events](https://neurips.cc/virtual/2025/affinity_events)

* * *

    *   [Socials](https://neurips.cc/virtual/2025/loc/san-diego/events/social)

* * *

    *   [Careers](https://neurips.cc/careers)

*   [Workshops](https://neurips.cc/virtual/2025/loc/san-diego/events/workshop)
*   [Exhibitors](https://neurips.cc/virtual/2025/sponsor_list)
*   [](https://neurips.cc/virtual/2025/search)
*   [Help](https://neurips.cc/virtual/2025/poster/117474#)
    *   [FAQ](https://neurips.cc/FAQ)

* * *

    *   [Organizers](https://neurips.cc/virtual/2025/organizers)

* * *

    *   [Help via Chat](https://neurips.cc/chat-directory)

*   [Expo](https://neurips.cc/virtual/2025/loc/san-diego/events/expo-2025)

Poster

# FAN: Fourier Analysis Networks

 Yihong Dong ⋅ Ge Li ⋅ Yongding Tao ⋅ Xue Jiang ⋅ Kechi Zhang ⋅ Jia Li ⋅ Jinliang Deng ⋅ Jing Su ⋅ Jun Zhang ⋅ Jingjing Xu 

2025 Poster

[Project Page](https://github.com/YihongDong/FAN) [[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/117474.png?t=1762774729.7636166 "Poster")] [[OpenReview](https://openreview.net/forum?id=Xpi0LpWbvF "OpenReview")]

### Abstract

Despite the remarkable successes of general-purpose neural networks, such as MLPs and Transformers, we find that they exhibit notable shortcomings in modeling and reasoning about periodic phenomena, achieving only marginal performance within the training domain and failing to generalize effectively to out-of-domain (OOD) scenarios. Periodicity is ubiquitous throughout nature and science. Therefore, neural networks should be equipped with the essential ability to model and handle periodicity. In this work, we propose FAN, a novel neural network that effectively addresses periodicity modeling challenges while offering broad applicability similar to MLP with fewer parameters and FLOPs. Periodicity is naturally integrated into FAN's structure and computational processes by introducing the Fourier Principle. Unlike existing Fourier-based networks, which possess particular periodicity modeling abilities but face challenges in scaling to deeper networks and are typically designed for specific tasks, our approach overcomes this challenge to enable scaling to large-scale models and maintains the capability to be applied to more types of tasks. Through extensive experiments, we demonstrate the superiority of FAN in periodicity modeling tasks and the effectiveness and generalizability of FAN across a range of real-world tasks. Moreover, we reveal that compared to existing Fourier-based networks, FAN accommodates both periodicity modeling and general-purpose modeling well.

Show more

### Video

Chat is not available.

Successful Page Load

NeurIPS uses cookies for essential functions only. We do not sell your personal information. [Our Privacy Policy »](https://neurips.cc/public/PrivacyPolicy)Accept

###### ![Image 6: NeurIPS logo](https://neurips.cc/static/core/img/NeurIPS-logo.svg)

The NeurIPS Logo above may be used on presentations. Right-click and choose download. It is a vector graphic and may be used at any scale.

###### Useful links

*   [Press](https://neurips.cc/Conferences/2025/Press)
*   [Proceedings](https://proceedings.neurips.cc/)

###### Contact

1269 Law St, San Diego CA 92109

[Email](https://neurips.cc/Help/Contact)

[NeurIPS Proceedings](https://proceedings.neurips.cc/)

[9] Title: Verifying your browser | OpenReview
[9] URL Source: https://openreview.net/forum?id=l4jBHP4FPy
[9] Description: 

## Complete the check below to continue to OpenReview

Please complete the verification above.

Have an OpenReview account? [Sign in](https://openreview.net/login?redirect=%2Fforum%3Fid%3Dl4jBHP4FPy) to skip this check.
