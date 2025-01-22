# SCALING RELATIONSHIP ON LEARNING MATHEMATICAL REASONING WITH LARGE LANGUAGE MODELS  

Zheng Yuan∗, Hongyi Yuan∗†, Chengpeng $\mathbf{Li}^{\dagger}$ , Guanting Dong†, Keming Lu Chuanqi Tan, Chang Zhou, Jingren Zhou  

Alibaba DAMO Academy {yuanzheng.yuanzhen,yuanhongyi.yhy}@alibaba-inc.com {lichengpeng.lcp,dongguanting.dgt,lukeming.lkm}@alibaba-inc.com chuanqi.tcq,ericzhou.zc,jingren.zhou @alibaba-inc.com  

# ABSTRACT  

Mathematical reasoning is a challenging task for large language models (LLMs), while the scaling relationship of it with respect to LLM capacity is under-explored. In this paper, we investigate how the pre-training loss, supervised data amount, and augmented data amount infuence the reasoning performances of a supervised LLM. We fnd that pre-training loss is a better indicator of the model’s performance than the model’s parameter count. We apply supervised fne-tuning (SFT) with different amounts of supervised data and empirically fnd a log-linear relation between data amount and model performance, and we fnd better models improve less with enlarged supervised datasets. To augment more data samples for improving model performances without any human effort, we propose to apply Rejection sampling Fine-Tuning (RFT). RFT uses supervised models to generate and collect correct reasoning paths as augmented fne-tuning datasets. We fnd with augmented samples containing more distinct reasoning paths, RFT improves mathematical reasoning performance more for LLMs. We also fnd RFT brings more improvement for less performant LLMs. Furthermore, we combine rejection samples from multiple models which push LLaMA-7B to an accuracy of $49.3\%$ on GSM8K which outperforms the supervised fne-tuning (SFT) accuracy of $35.9\%$ signifcantly. We release our codes and rejection sampling augmented data in https://github.com/OFA-Sys/gsm8k-ScRel.  

# 1 INTRODUCTION  

Large language models (LLMs) (Anil et al., 2023; Touvron et al., 2023b; OpenAI, 2023) have shown considerable abilities in various math reasoning tasks (Saxton et al., 2019; Cobbe et al., 2021; Lightman et al., 2023). It is of interest to understand, predict, and improve an LLM’s math reasoning ability based on different pre-trained LLMs and supervised datasets. With this knowledge, we can better decide the effort we put into improving the LLM or augmenting the dataset. Many recent works are focusing on using different prompts (Wei et al., 2022b; Yao et al., 2023) or ensembling / reranking multiple times of inferences (Cobbe et al., 2021; Uesato et al., 2022; Wang et al., 2023; Lightman et al., 2023) to improve models’ reasoning performances. While in-context learning (ICL) and performing multiple inferences can improve performance, it is computationally expensive and not suitable for online deployment scenarios. Therefore, we focus on the performance of the supervised LLMs with inference only once which is a setting closer to online deployment.  

To this end, we empirically investigate the scaling relationship of factors that infuence the math reasoning abilities of a supervised LLM, including pre-training losses, the amount of supervised data, and the amount of augmented data. Firstly, we analyze the supervised fne-tuning (SFT) and ICL performance of LLMs. We observe that the pre-training loss is approximately negatively linear correlated to the SFT and ICL accuracy in a given interval which is a better performance indicator than pre-trained model sizes or pre-trained token counts. Secondly, we analyze the relationship between SFT and different amounts of supervised data. We observe that the model performance has a log-linear relation versus the supervised data amount while the increase diminishes with the better pre-trained model. Thirdly, we want to leverage the model itself to generate more supervised data to reinforce its reasoning ability and analyze the scaling relationship of the augmented data amount. We apply rejection sampling on SFT models to sample and select correct reasoning paths as augmented dataset (Uesato et al., 2022; Zhu et al., 2023). We use these augmented datasets to fne-tune base LLMs which would achieve better performances compared to SFT and we denote it as rejection sampling fne-tuning (RFT). We fnd the key factor infuencing RFT performance is the distinct reasoning path amount which can be increased by sampling more times or combing samples from multiple models. We apply RFT on several pre-trained LLMs and show larger improvement on less performant models. We discuss the reason RFT works is it provides multiple reasoning paths which makes LLMs have better reasoning generalization. We also discuss that RFT is much cheaper than pre-training in computational resources while training an LLM with lower pre-training loss is the fundamental solution.  

![](https://cdn-mineru.openxlab.org.cn/extract/8432c534-d1a8-4d5b-b8b0-0fefd429f45a/16ed41c30dbc1d3159b2a67ec6aead7f0a9702357a575c403e6f6efbbfaf4659.jpg)  
Figure 1: The key fndings of scaling relationship on learning math reasoning ability with LLMs.  

The key fndings of this paper are shown in Figure 1 and are summarized here:  

• When the pre-training loss gets smaller (i.e. the pre-trained model gets better), the model reasoning performances of SFT and ICL increase linearly within a range. The SFT performance improves slower than ICL.   
• SFT improves in a log-linear manner with the increase of supervised data amount. The benefts of increasing data amount diminish as the pre-trained model gets better.   
• The model performance for RFT improves as the distinct reasoning path amount increases. The RFT performance improves slower than SFT.   
• The combination of rejection sampling samples from multiple models further enhances the RFT performance, resulting in an accuracy of 49.3 for LLaMA-7B $_{\cdot+13.4}$ compared to SFT), 50.3 for LLaMA2-7B $_{\left.+8.7\right.}$ compared to SFT), 52.1 for LLaMA-13B $_{\leftmoon}+9.1$ compared to SFT), and 55.4 for LLaMA2-13B $_{\left.+5.4\right.}$ compared to SFT).  

# 2 RELATED WORKS  

Learning Math Reasoning with LLMs Recent research on LLMs has discovered the emergent ability to solve reasoning tasks beyond a certain model scale (Wei et al., 2022a). Such reasoning abilities in LLMs can be elicited by fne-tuning, few-shot prompting, or zero-shot prompting (Cobbe et al., 2021; Wei et al., 2021; Nye et al., 2021; Wei et al., 2022b; Kojima et al., 2022). A large amount of research focuses on the reasoning tasks of math word problems (MWP), and methods are evaluated on the benchmarks spanning different levels of MWPs (Koncel-Kedziorski et al. (2016); Patel et al. (2021); Lan et al. (2021); Cobbe et al. (2021); Jie et al. (2022); Yuan et al. (2023a); Fu et al. (2023a), inter alia). The core idea of improving the mathematical reasoning ability of LLMs is to aggregate various sampled reasoning paths during either fne-tuning or inference. Cobbe et al. (2021) trained and devised a reasoning path verifer to select the correct results during inference. Wang et al. (2023) proposed to sample various reasoning paths during inference and then derive the fnal result by majority voting on the answers or through verifers (Li et al., 2023). Several works applied the idea of rejection sampling along with other techniques to flter the diverse sampled reasoning paths for fne-tuning data augmentation (Huang et al., 2022; Zelikman et al., 2022; Ni et al., 2023; Zhu et al., 2023). Rejection sampling is a simple-yet-effective fne-tuning augmentation technique and is also used for LLM alignment with human preference (Bai et al., 2022; Yuan et al., 2023b; Dong et al., 2023; Touvron et al., 2023b; Song et al., 2023). Uesato et al. (2022) explored to use of reinforcement learning methods for improving the mathematical reasoning abilities of LLMs and they further discussed the difference between outcome-based and process-based reward modeling. Followed by Lightman et al. (2023), they collected large-scale process-based supervision signals through human annotation and verifed that LLMs can beneft more from process-based reward modeling with human-annotated supervision than outcome-based reward modeling. There is also prior research that distilled the emergent reasoning ability of LLMs to small language models (Fu et al., 2023b; Shridhar et al., 2023). Compared to previous works (Zelikman et al., 2022; Uesato et al., 2022; Zhu et al., 2023; Ni et al., 2023), we are using a simpler way of generating augmented samples without any trained process-level reward models and we are focusing on researching the scaling relationship between LLMs and math reasoning ability.  

Scaling Laws of Large Language Models It is important to understand and predict the performance gain as the language model scales up. Kaplan et al. (2020) frst investigated and derived a predictable relationship on how the number of model parameters and data sizes contribute to the loss over many orders of magnitudes. Hoffmann et al. (2022) refned the scaling laws in (Kaplan et al., 2020) and found the scaling laws for computation-optimal training. Muennighoff et al. (2023) explored and extended the scaling laws under a data-constrained scenario. Besides investigating the scaling performance for pre-training, Gao et al. (2022) discussed the scaling laws for overparameterized reward models for alignment with human preference, and Hernandez et al. (2021) developed scaling laws for transferring performance from pre-trained models to downstream tasks. Henighan et al. (2020); Caballero et al. (2022) investigated scaling laws of math problems. In this paper, we are investigating the scaling relationships of large language models on learning math word problems with pre-training losses, supervised data amount, and augmented data amount.  

# 3 THE FACTORS OF MATH REASONING ABILITY IN SUPERVISED LLM  

The target of this paper is to try to understand the performances of supervised LLMs in math reasoning. We expect a pre-trained LLM $\rho$ to learn reasoning ability from a supervised reasoning dataset $\mathcal{D}$ . The dataset is defned by $\boldsymbol{D}=\{q_{i},r_{i},a_{i}\}_{i}$ , where $q$ is a question, $r$ is a chain-of-thought reasoning path, and $a$ is a numerical answer. We perform supervised fne-tuning on dataset $\mathcal{D}$ to obtain an SFT model $\pi$ . We use $\pi$ to generate reasoning paths and answers in the test set by greedy decoding and report the accuracy (i.e. acc or maj1 $@1$ ) as our metric here.  

# 3.1 MODEL ACCURACY VS. PRE-TRAINING LOSS  

Previous works state that the larger LLM shows better reasoning ability across the same series of models (Brown et al., 2020; Chowdhery et al., 2022; Touvron et al., 2023a;b), and we fnd LLaMA outperforms GPT-3 which shows the model parameter counts should not be the only indicator of reasoning ability. While LLMs have different architectures, model parameters, and pre-training token numbers, we fnd the pre-training loss is a stable performance indicator of the math reasoning ability and we use it to represent the model instead of using their model parameters and pre-training token numbers.  

We analyze the SFT and ICL (8-shot) performance of GPT-3 (Brown et al., 2020), LLaMA (Touvron et al., 2023a), LLaMA2 (Touvron et al., 2023b), and GPT-4 (OpenAI, 2023). The pre-training losses of these models are observed in their paper, we should notice that pre-training losses correspond to different pre-training datasets and different tokenizers which means they could not be compared strictly (and we cannot use it to do any sort of regression directly) while the tendency among these losses is still enlightening. We use the results of GPT-3 fne-tuning from (Cobbe et al., 2021) and we fne-tune LLaMA and LLaMA2 on the GSM8K training set (detailed in Appendix A.1). For in-context learning, we use the results from LLaMA (Touvron et al., 2023a) and LLaMA2 (Touvron et al., 2023b) paper.  

![](https://cdn-mineru.openxlab.org.cn/extract/8432c534-d1a8-4d5b-b8b0-0fefd429f45a/2d37aedaa03ba654872287e89dad08fca37130e5930b3e32971d30d162a716a7.jpg)  
Figure 2: The performance of SFT (blue lines) and ICL (red lines) settings on GSM8K. GPT-4 states they use some part of the GSM8K data in pre-training, and suggest others consider its performance between SFT and ICL.  

In Figure 2, we can fnd that:  

• The pre-training losses are approximately negatively linear correlated to the SFT and ICL accuracy during the given pre-training loss interval.   
• SFT outperforms ICL consistently, while the improvements diminish when the pre-training loss is lower.  

The linear relation of SFT and ICL accuracy may only work in the given interval. The reasons are (1) the slope of ICL is steeper than SFT, while the SFT performance should be greater than ICL performance; (2) the accuracy can not bigger than 1 or smaller than 0. It should be using $-\log(a c c)$ instead of acc as the dependent variable theoretically while we fnd an apparent linear relationship among pre-training loss and acc and use acc as the dependent variable. LLaMA-2 7B(13B) can be viewed as an approximation of continue-training of LLaMA 7B(13B). As it trains longer, its ICL and SFT performance both improve without changing the parameter count. From the observations, one effective way to improve reasoning ability is to train a better base model with lower pre-training loss (Pre-training is all you need!). The models with lower pre-training loss improve less from the fne-tuning which may be due to the models having already obtained more reasoning abilities during pre-training and the supervised data can provide less signal to supervise them.  

![](https://cdn-mineru.openxlab.org.cn/extract/8432c534-d1a8-4d5b-b8b0-0fefd429f45a/23af877b13c9911febc1032753c08065b57a02293f559f0897471815fd40ed6b.jpg)  
Figure 3: The performance of SFT with different amounts of supervised data on GSM8K.  

# 3.2 MODEL ACCURACY VS. SUPERVISED DATA COUNT  

Supervised fne-tuning does improve LLMs’ reasoning ability, we want to know how the supervised data amount infuences the model’s improvement. We fne-tune LLaMA and LLaMA2 with $\{1,1/2,1/4,1/8,1/16,1/32\}$ amount of the training set from GSM8K (detailed in Appendix A.2). We want to use this experiment to extrapolate the model performances if we have more supervised data. In Figure 3, we plot the results of training with different amounts of supervised data. From this fgure, we can observe that:  

• The model performance has a log-linear relation versus data amount. When the data amount doubles, the performance increases by a unit.   
• Better model needs more amount of data to outperform its ICL performance.   
• Better model benefts less when supervised data amount doubles.  

The log-linear relation is stable during $\{1,1/2,1/4,1/8\}$ amount of the training data. From the observation, it is straightforward to enlarge the training dataset to improve the performance, especially for worse models. For better models, it benefts less which echoes that better models have learned more reasoning ability during pre-training.  

# 3.3 MODEL ACCURACY VS. AUGMENTED DATA COUNT  

Increasing the amount of math reasoning labeled data is diffcult, especially proposing a new question. It is easy for a well-educated student to solve hundreds of math word problems per day, but it is very hard to come up with diverse and educational math problems. So our direction changes to augment new data using existing resources. We have tried augmenting new queries (detailed in Appendix D.1) and augmenting revisions (detailed in Appendix D.2). These approaches have none to marginal improvements compared to SFT. We fnd a simplifed version of rejection sampling (Zhu et al., 2023) is a naive and effective way to augment new reasoning paths and can improve the model performance. And we fnd the key factor infuences fne-tuning on rejection sampling (RFT) augmented data is distinct reasoning path amount. Combining rejection sampling samples from multiple models, we can further fne-tune a LLaMA-7B model to an accuracy of 49.3 (compared with SFT 35.9) and a LLaMA-13B model to an accuracy of 52.1 (compared with SFT 43.0).  

Table 1: The performance of RFT with $k=100$ on GSM8K compared with SFT and ICL. Distinct path amount means distinct equation list amount here.   


<html><body><table><tr><td>Setting</td><td>7B</td><td>7B-2</td><td>13B</td><td>13B-2</td><td>33B</td></tr><tr><td>Pretrainloss</td><td>1.8</td><td>1.75</td><td>1.73</td><td>1.68</td><td>1.62</td></tr><tr><td>ICL SFT</td><td>11.0/18.1 35.9/48.7</td><td>14.6/- 41.6/55.4</td><td>17.8/29.3 43.0/55.2</td><td>28.7/- 50.0/61.7</td><td>35.6/53.1 54.6/-</td></tr><tr><td>RFTk=100</td><td>41.7/52.7</td><td>47.5/58.7</td><td>49.1/59.9</td><td>54.8/65.4</td><td>54.5/-</td></tr><tr><td>Correct paths per question</td><td>53.3</td><td>60.8</td><td>62.5</td><td>71.6</td><td>88.7</td></tr><tr><td>Distinct paths per question</td><td>5.25</td><td>5.19</td><td>5.26</td><td>5.29</td><td>2.78</td></tr></table></body></html>  

Rejection Sampling Fine-tuning The SFT model $\pi$ obtains the ability to perform zero-shot chainof-thought reasoning, and we use $\pi$ to generate more correct reasoning paths $r_{i j}$ to supply the training dataset. For each $q_{i}$ , we generate $k$ candidate reasoning paths and answers $r,a$ with a temperature of 0.7 following (Cobbe et al., 2021). We frst flter out reasoning paths with wrong answers $a\neq a_{i}$ or wrong calculations based on Python evaluation. Each reasoning path contains a list of equations $e_{j}$ , and we select one reasoning path $r_{i j}$ for each distinct equation list as the augmented data and remove other reasoning paths with the same list of equations to deduplicate similar reasoning paths. Different order of elements (e.g. $3+4=7$ and $4+3=7$ ) or different order of equations (e.g. $1+2=3,3+4=7$ and $1+4=5,2+5=7)$ are considered different. It is helpful for models to know these orders can be exchanged and is hard for models to learn this with only one reasoning path each problem. We defne $\mathcal{D}_{\pi}^{\prime}=\mathcal{D}\cup\{q_{i},r_{i j},a_{i}\}_{i,j}$ as the augmented dataset. We fne-tune $\mathcal{D}^{\prime}$ on pre-trained LLM $\rho$ to $\pi_{\mathrm{RFT}}$ as RFT, and we detail how we apply RFT in Appendix A.3. We list the results of RFT with sampling $k\,=\,100$ candidate reasoning paths on LLaMA and LLaMA-2 in Table 1. For ICL, SFT, and RFT, we list the maj1 $@1$ (accuracy) and maj1 $@100$ (sample 100 times and calculate accuracy based on majority voting) as metrics.  

In the case of 7B and 13B models, RFT yields an approximate increase of 5 to 6 points in maj $\lfloor\@\,1$ and about 4 points increase in maj1 $@100$ . For 33B models, RFT does not improve performance compared to SFT. The main reason comes from the augmented samples from rejection sampling. We can fnd that better models generate more correct reasoning paths per question. For LLaMA33B-SFT, it can generate an average of 88.7 correct paths per question. However, it overfts the training set and has diffculty generating more diverse paths on the training set questions. Rejection sampling with 33B is very time-consuming and we do not conduct a temperate grid search, we have tried using a larger temperate 1.0 for decoding LLaMA-33B-SFT models, it generates 82.4 correct paths and 4.77 distinct paths per question which is more diverse than using temperate 0.7 but still less diverse than 7B and 13B models. We admit there should be a temperate (or generation confg) that can produce more distinct paths and generate good results for RFT in 33B and even larger models while it does need more computation resources for inference compared to sampling using 7B and 13B models. We will show we can use 7B and 13B models only for rejection sampling to improve the 33B model.  

Model Accuracy vs Rejection Sampling Data Count To understand the performance of RFT, we vary $k$ among 1, 3, 6, 12, 25, 50, 100 and apply RFT. We also have another setting of $k=100$ while not removing any reasoning paths denoted as no dedup. We list the RFT results with different $k$ on Figure 4. Comparing using RFT with $k=100$ and no dedup, the performance is similar and shows that it is better to estimate RFT performance based on distinct reasoning path amount instead of RFT augmented sample counts. Furthermore, using deduplication has better performances for 3 of 4 models and needs much less training time.  

When using $k\,=\,3$ , RFT outperforms SFT by 2 points stably. For most data points, using larger $k$ leads to better performances. However, the merits of RFT are decreasing when doubling $k$ . We calculate different paths per question for different $k$ in Table 2. We can see that the amount of different reasoning paths is not growing quickly along $k$ growing. In Figure 3, we know doubling training samples can have a linear performance improvement. Doubling reasoning paths should improve less than doubling training samples since obtaining different reasoning paths does not obtain any new questions. Therefore, doubling $k$ leads to diminished performance improvements.  

![](https://cdn-mineru.openxlab.org.cn/extract/8432c534-d1a8-4d5b-b8b0-0fefd429f45a/c41ea161c46f8b6458e4270724ea3c437602efb749d854329554ecc4679e6dc3.jpg)  
Figure 4: The performance of RFT with different amounts of sampling count $k$ on GSM8K.  

rent reasoning paths per question generated by different SFT models with   


<html><body><table><tr><td>k</td><td>7B</td><td>7B-2</td><td>13B</td><td>13B-2</td><td>33B</td></tr><tr><td>1 3 6 12 25 50</td><td>1.17 1.44 1.74 2.20 2.93 3.94</td><td>1.19 1.47 1.78 2.23 2.93 3.91</td><td>1.15 1.41 1.69 2.11 2.88 3.90</td><td>1.18 1.45 1.76 2.21 2.94 3.94</td><td>1.06 1.16 1.28 1.46 1.77</td></tr><tr><td>100 400(U13B) 500(U33B）</td><td>5.25</td><td>5.19</td><td>5.26 12.84 13.65</td><td>5.29</td><td>2.19 2.78</td></tr></table></body></html>  

Combining rejection sampling samples from multiple models The experiment results above demonstrate performance boosts in mathematical reasoning, beneftting from rejection sampling. Through case studies in 4.1, we show that rejection sampling can augment training data with reasoning paths of diverse calculation processes. However, the reasoning paths sampled from one single SFT model can be logically non-diverse. Therefore, we expect to further improve the mathematical reasoning performance by leveraging rejection sampled reasoning paths aggregated from different models. We denote two fnal datasets as $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ and $\mathcal{D}_{\mathrm{U33B}}^{\prime}$ , which are aggregated from rejection sampling different models $\mathcal{D}_{\mathrm{U13B}}^{\prime}=\mathcal{D}_{7\mathrm{B}}^{\prime}\oplus\mathcal{D}_{7\mathrm{B}2}^{\prime}\oplus\mathcal{D}_{13\mathrm{B}}^{\prime}\oplus\bar{\mathcal{D}}_{13\mathrm{B}2}^{\prime}$ and $\mathcal{D}_{\mathrm{U33B}}^{\prime}=\mathcal{D}_{\mathrm{U13B}}^{\prime}\oplus\mathcal{D}_{33\mathrm{B}}^{\prime}$ , where U means models under a certain size, 7B/13B/33B means LLaMA-7B/13B/33B and 7B2/13B2 means LLaMA2-7B/13B. $\bigoplus$ means an aggregation process in which all the reasoning paths from different sets are frst combined and then Algorithm 1 is applied to deduplicate the reasoning paths with the same calculation process regarding the equation forms and orders.  

We can see, through the results visualized in Figure 5, that using the aggregated dataset $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ and $\mathcal{D}_{\mathrm{U33B}}^{\prime}$ can lead to uniformly better performance than fne-tuning with datasets from a single model across different model sizes. RFT on these two augmented datasets $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ and $\mathcal{D}_{\mathrm{U33B}}^{\prime}$ decreases the performance gaps among the same size models in SFT and RFT $k=100$ which mean the combined augmented datasets provide enough reasoning supervision to fulfll the pre-training gap. We can assume with suffcient supervised data amounts, the performance indicator should be the model size but not the pre-training losses.  

![](https://cdn-mineru.openxlab.org.cn/extract/8432c534-d1a8-4d5b-b8b0-0fefd429f45a/57bff5edc7677be624135ee38a89492c9a265739268c5dbbb97fc190c0f3f71b.jpg)  
Figure 5: The performance of RFT with rejection sampling samples from multiple models.  

We have stated that it is expensive to apply RFT $k=100$ on 33B models and it needs a temperate grid search to achieve an improvement compared to SFT. However fne-tuning on $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ has similar rejection sampling computational cost compared with sampling 100 times on 33B and achieve better performance.  

Another phenomenon is including $\mathcal{D}_{338}^{\prime}$ in aggregation barely infuences the performance. To give a more comprehensive analysis of the results, we calculate the average reasoning path number per question in Table 2 and depict a Venn diagram to visualize the source of different reasoning paths shown in Figure 6. In Table 2, the average reasoning path numbers of $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ and $\mathcal{D}_{\mathrm{U33B}}^{\prime}$ surpass those of a single model by large amounts, while $\mathcal{D}_{\mathrm{U33B}}^{\prime}$ only have slightly more reasoning paths than $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ by 0.81. In the meanwhile, as shown in Figure 6, the models under and including the size of 13B can contribute unique reasoning paths of similar proportion in $\mathcal{D}_{\mathrm{U33B}}^{\prime}$ around $15\%$ . However, only $6.5\%$ of the reasoning paths can be exclusively acquired from LLaMA-33B-SFT model. This shows that the SFT model of 33B can provide limited reasoning diversity when sampling the training questions. This fnding is consistent with the results above in Table 1, indicating the 33B model (and possibly 65B and 70B models) can well memorize the human-annotated reasoning paths.  

For 65B models, we fnd using $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ does not improve the performance compared to SFT. The reason can be better models beneft less from the supervised sample amounts while it has learnt more reasoning ability during pre-training.  

Overall, we can come to the conclusion that (1) RFT improves the mathematical reasoning performance of (worse) LLMs through diverse reasoning paths from rejection sampling of the SFT models, and aggregating more diverse reasoning paths can improve the performance further. (2) Different SFT models can contribute reasoning paths with different calculation processes from rejection sampling, leading to more diverse training data for RFT, and LLMs of larger parameter sizes may degrade in generating diversifed reasoning paths as a result of overftting the training questions. There may be a generation confg or training confg for large enough LMs not to overft on the training dataset while it is not trivial to fnd them.  

Comparing to other baselines We compare our RFT results of training on $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ to several baselines and the results are detailed in Table 3. Although LLaMA and LLaMA2 are top-tier opensourced LLMs 1, their mathematical reasoning performances still lag behind the current proprietary LLMs which are of larger parameter scales, such as GPT-4 and PaLM2. Compared to results on  

![](https://cdn-mineru.openxlab.org.cn/extract/8432c534-d1a8-4d5b-b8b0-0fefd429f45a/3a65086c30ce32eb81f347445226e722cf10774966832e96f34fafa76877c9d8.jpg)  

Figure 6: The Venn diagram of the proportions of the reasoning calculation paths that each model provide to $\mathcal{D}_{\mathrm{U33B}}^{\prime}$ . For example, $15.5\%$ (in the yellow part) of the reasoning calculation paths in $\mathcal{D}_{\mathrm{U33B}}^{\prime}$ can only be exclusively found in the rejection sampling results from LLaMA2-13B-SFT.  

open-resourced models, our results on LLaMA present better performance than two recent stateof-the-art reasoning augmentation methods. Our RFT method is simpler compared to CoRE, since RFT does not require training verifer models and decoding with Monte Carlo Tree Search (MCTS). Compared to other open-sourced aligned language models, we can fnd that 7B models struggle at a level of 35 scores which are very similar to SFT performances of LLaMA-7B. We guess they use GSM8K during their pre-training phase following (OpenAI, 2023) or human alignment fne-tuning phase following (Qingyi et al., 2023). Using our augmented dataset $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ to replace the original GSM8K can signifcantly boost their 7B models’ performances.  

# 4 DISCUSSION  

# 4.1 DIFFERENT DISTRIBUTION OF REASONING PATHS  

In the aforementioned analysis of RFT training data, we observe that rejection sampling can augment the training question with diverse reasoning calculation paths. In this section, we investigate whether RFT models can learn to generate different reasoning paths to reach the correct answers. We fnetune LLaMA and LLaMA2 of 7B and 13B on $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ . During inference, we sample 100 different reasoning paths from each trained model for each test set question with a temperature of 0.7. For each question, we compute the number of different calculation processes presented in 100 sampled reasoning paths that lead to the correct answer and draw histograms with respect to test set questions. SFT and RFT models on self-sampled datasets (RFT $k{=}100$ ) are included for comparison.  

As shown in Figure 7, the models trained by RFT on $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ exhibit more question counts than the models trained by RFT $k{=}100$ and SFT on the larger numbers of unique calculation processes. There are more question counts for SFT models where all the sampled reasoning paths only correspond to one single calculation process and SFT models can barely generate more than 8 different calculation processes for a question. This analysis demonstrates that diverse reasoning calculation paths in training data can equip the LLMs with fnding diverse reasoning logic for solving math problems.  

<html><body><table><tr><td>Base Model</td><td>Training</td><td>maj1@1</td><td>maj1@K*</td></tr><tr><td>Proprietary LLMs GPT-4 (OpenAI, 2023)</td><td>5-shot ICL</td><td>92.0 34.0</td><td></td></tr><tr><td>GPT-3-175B (Brown et al.,2020) PaLM2 (Anil et al., 2023) PaLM-540B (Chowdhery et al.,2022)</td><td>SFT 8-shot ICL 8-shot ICL</td><td>80.7</td><td>91.0@K=40</td></tr><tr><td>Chinchilla-70B (Uesato et al.,2022)</td><td>5-shot ICL</td><td>56.5 43.7 58.9</td><td>74.4@K=40 58.6@K=96 77.7@K=96</td></tr><tr><td>Chinchilla-70B Open-sourced LLMs</td><td>SFT</td><td></td><td>41.4</td></tr><tr><td>GPT-Neo-2.7B (Black et al., 2021) GPT-J-6B (Wang & Komatsuzaki, 2021)</td><td>FCS + PCS (Ni et al., 2023) CoRE (Zhu et al.,2023)</td><td>19.5</td><td></td></tr><tr><td>ChatGLM2-6B (Zeng et al., 2022)</td><td>8-shot ICL</td><td>34.9 32.4</td><td>63.2@K=40</td></tr><tr><td>ChatGLM2-6B ChatGLM2-12B</td><td>Human Alignment 8-shot ICL</td><td>28.1 40.9</td><td></td></tr><tr><td>ChatGLM2-12B</td><td>Human Alignment</td><td>38.1</td><td></td></tr><tr><td>InternLM-7B (Team, 2023)</td><td>4-shot ICL</td><td>31.2</td><td></td></tr><tr><td>InternLM-7B</td><td>Human Alignment</td><td>34.5</td><td></td></tr><tr><td>LLaMA-7B</td><td>SFT</td><td>35.9</td><td></td></tr><tr><td></td><td></td><td></td><td>48.7</td></tr><tr><td>Our RFT on open-sourced LLMs</td><td>RFT-U13B</td><td>49.3</td><td></td></tr><tr><td>LLaMA-7B</td><td></td><td></td><td>61.8</td></tr><tr><td>LLaMA2-7B</td><td>RFT-U13B</td><td></td><td></td></tr><tr><td>LLaMA-13B</td><td>RFT-U13B</td><td>50.3 52.1</td><td>65.6</td></tr><tr><td></td><td>RFT-U13B</td><td></td><td>66.2 69.1</td></tr><tr><td>LLaMA2-13B</td><td></td><td>55.4</td><td></td></tr></table></body></html>  

![](https://cdn-mineru.openxlab.org.cn/extract/8432c534-d1a8-4d5b-b8b0-0fefd429f45a/ea34161555a1505472ed32aeb957fd3794b8f266b26fb01305c435a7889fe294.jpg)  
Table 3: Compare GSM8K results with other baselines. RFT-U13B means models fne-tuned on $\mathcal{D}_{\mathrm{U13B}}^{\prime}$ . FCS and PCS represent fully-correct solutions and partially-correct solutions respectively. $\!\!\!\nabla\!\!\!\cdot\!\!\!\!\in\!\!\!\!100$ if not specifed.   
Figure 7: The histograms of question numbers solved with different numbers of unique reasoning calculation paths. We show the difference in question counts between SFT and RFT U13B in two cases where the numbers of unique reasoning calculation paths are 1 or more than 10.  

<html><body><table><tr><td>Model size</td><td>7B</td><td>7B-2</td><td>13B</td><td>13B-2</td><td>33B</td><td>65B</td><td>70B</td></tr><tr><td>Pre-train FLOPs</td><td>4.2 × 1022</td><td>8.4 × 1022</td><td>7.8 × 1022</td><td>1.6 × 1023</td><td>2.7 × 1023</td><td>5.5 × 1023</td><td>8.4 × 1023</td></tr><tr><td>SFT FLOPs</td><td>1.7 × 1017</td><td></td><td>3.3 × 1017</td><td></td><td>7.7 × 1017</td><td>1.3 × 1018</td><td>1.7 × 1018</td></tr><tr><td>RFT Inference FLOPs</td><td>1.4 × 1018</td><td></td><td>2.6 × 1018</td><td></td><td>6.9 × 1018</td><td>1.4 × 1019</td><td>1.8 × 1019</td></tr><tr><td>RFT-U33BFLOPs</td><td>3.0 × 1018</td><td></td><td>5.7 × 1018</td><td></td><td>1.3 × 1019</td><td>2.2 × 1019</td><td>3.0 × 1019</td></tr><tr><td>Pre-train GPU hrs</td><td>82k</td><td>184k</td><td>135k</td><td>368k</td><td>530k</td><td>1022k</td><td>1720k</td></tr><tr><td>SFT GPU hrs</td><td>0.6</td><td></td><td>4</td><td></td><td>40</td><td>74</td><td>80</td></tr><tr><td>RFT Inference GPU hrs</td><td>10</td><td></td><td>0.1k</td><td></td><td>0.1k</td><td>4.3k</td><td>4.5k</td></tr><tr><td>RFT-U33B GPU hrs</td><td>6</td><td></td><td>62</td><td></td><td>0.6k</td><td>1k</td><td>1.2k</td></tr><tr><td>ICL Accuracy</td><td>11.0</td><td>14.6</td><td>17.8</td><td>28.7</td><td>35.6</td><td>50.9</td><td>56.8</td></tr><tr><td>SFT Accuracy</td><td>35.9</td><td>41.6</td><td>43.0</td><td>50.0</td><td>54.6</td><td>59.3</td><td>63.2</td></tr><tr><td>RFT-U33BAccuracy</td><td>49.1</td><td>51.2</td><td>51.4</td><td>55.3</td><td>57.9</td><td></td><td></td></tr></table></body></html>  

Table 4: The statistics of FLOPs and GPU hours required for pre-training, SFT, RFT inference, and RFT. We take the pre-training GPU hours from Touvron et al. (2023a;b). The GPU hours for RFT inference are calculated for 7,473 train set questions and 100 samples per question. To make the best of GPUs and properly ft models into the GPU memory, we tune the inference batch size. For 33B, 65B, and 70B models, we use DeepSpeed ZeRO3 (Rasley et al., 2020) for distributed training. All the GPU hours are based on NVIDIA A100 80GB GPU. Note we use non-embedding parameters to compute FLOPs in our experiments.  

# 4.2 TOWARDS EXCELSIOR MATHEMATICAL REASONING  

From our fndings, there are two main factors that can improve mathematical reasoning abilities given a preset amount of human-annotated samples, including: (1) Pre-training the LLMs to lower losses; (2) Augmenting fne-tuning with rejection sampling. Through extensive experiments, we empirically verify the scaling relationships between the mathematical reasoning performance of LLM with both factors respectively. Out of the consideration of sustainable NLP, in this section, we investigate the possible computational resources required to extrapolate the mathematical performance of LLMs by both factors and discuss how to improve the performance more effciently.  

We estimate the pre-training, SFT, RFT inference, and RFT FLOPs following Kaplan et al. (2020) and GPU times in Table 4 which is detailed in Appendix E. We can fnd that the cost times of SFT $(\sim1\times10^{-5})$ and RFT $(\sim1\times10^{-4})$ are negligible compared to pre-training. One can always use SFT and RFT to improve models’ performance. However, it could be hard to use RFT to further boost performance. Since we need much more sampling counts (at an exponential level) to increase distinct reasoning paths and there exists an upper bound of distinct reasoning path amount for a given math reasoning question.  

We assume that performance follows $\scriptstyle\mathrm{RFT}>\mathrm{SFT}>\mathrm{ICL}$ , from the fndings in this paper we know the improvement speed follows $_\mathrm{RFT<SFT<ICL}$ . And if we have an omnipotent language model which has a pre-training loss that is the same as the corpus randomness, it could have $\mathrm{RFT}=\mathrm{SFT}$ $=\mathrm{ICL}=100$ . Thus when you pre-train a better language model (i.e. smaller pre-training loss), your model’s performance still follows $\scriptstyle\mathrm{RFT}>\mathrm{SFT}>\mathrm{ICL}$ but their performance gaps are diminishing. Since you can obtain an RFT model without too much effort (compared to pre-training), then the most important thing we should do is to decrease the model’s pre-training loss. From LLaMA-7B to LLaMA2-7B, it needs to add $4.2\!\times\!10^{22}$ FLOPs to obtain a 2.1 improvement in the RFT-U33B setting with a 0.05 pre-training loss decrease. From LLaMA-7B to LLaMA-13B, it adds $3.6\times10^{22}$ FLOPs to obtain a 2.3 improvement in the RFT-U33B setting with a 0.07 pre-training loss decrease. While minimizing pre-training loss is expensive compared to SFT and RFT, we believe other abilities may follow a similar pattern and better pre-training can beneft all other tasks.  

# 5 CONCLUSIONS  

In this paper, we are investigating the scaling relationship in supervising math reasoning abilities with large language models. We fnd the relationship between math performance and pre-training  

losses, supervised data amount, and distinct reasoning paths. We fnd that better language models beneft less with SFT and RFT, and the most important thing is to pre-train a better language model towards excellent math reasoning abilities.  

# 6 ACKNOWLEDGEMENT  

We would like to express our sincere appreciation to Tianhang Zhu, Runji Lin, Kai Dang, Keming Lu, Wei Wang, and Junyang Lin for their valuable insights and contributions to this paper.  

# 7 LIMITATIONS  

In this paper, we miss the following parts which are very important for building math reasoning abilities for LLMs and should be discussed in the revised version of this paper or future works.  

• RFT for 65B and 70B LLaMA models.   
• Pre-training on the math-related corpus. This is obviously useful shown in Lewkowycz et al. (2022). While the pre-training loss obtained here cannot align with general domain pre-trained models’ losses.   
• We do not regress any scaling laws in this paper since many numbers are estimated and pre-training losses, ICL prompts and SFT settings of various models may not be aligned.  

# REFERENCES  

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/ paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf.  

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.  

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli TranJohnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfeld-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback, 2022.  

Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorfow, March 2021. URL https://doi. org/10.5281/zenodo.5297715.  

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. ArXiv, abs/2005.14165, 2020. URL https://api.semanticscholar.org/ CorpusID:218971783.  

Ethan Caballero, Kshitij Gupta, Irina Rish, and David Krueger. Broken neural scaling laws. arXiv preprint arXiv:2210.14891, 2022.  

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022.  

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.  

Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked fnetuning for generative foundation model alignment, 2023.  

Yao Fu, Litu Ou, Mingyu Chen, Yuhao Wan, Hao Peng, and Tushar Khot. Chain-of-thought hub: A continuous effort to measure large language models’ reasoning performance, 2023a.  

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. Specializing smaller language models towards multi-step reasoning. arXiv preprint arXiv:2301.12726, 2023b.  

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization, 2022.  

Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.  

Danny Hernandez, Jared Kaplan, Tom Henighan, and Sam McCandlish. Scaling laws for transfer, 2021.  

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022.  

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve, 2022.  

Zhanming Jie, Jierui Li, and Wei Lu. Learning to reason deductively: Math word problem solving as complex relation extraction. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5944–5955, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.410. URL https://aclanthology.org/2022.acl-long.410.  

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. CoRR, abs/2001.08361, 2020. URL https://arxiv.org/abs/2001.08361.  

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=e2TBb5y0yFf.  

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS: A math word problem repository. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1152–1157, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/N16-1136. URL https://aclanthology.org/N16-1136.  

Yihuai Lan, Lei Wang, Qiyuan Zhang, Yunshi Lan, Bing Tian Dai, Yan Wang, Dongxiang Zhang, and Ee-Peng Lim. Mwptoolkit: An open-source framework for deep learning-based math word problem solvers, 2021.  

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022.  

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifer. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5315– 5333, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https: //aclanthology.org/2023.acl-long.291.  

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.  

Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models, 2023.  

Ansong Ni, Jeevana Priya Inala, Chenglong Wang, Alex Polozov, Christopher Meek, Dragomir Radev, and Jianfeng Gao. Learning math reasoning from self-sampled correct and partiallycorrect solutions. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id $=$ 4D4TSJE6-K.  

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: Scratchpads for intermediate computation with language models, 2021.  

OpenAI. Gpt-4 technical report, 2023.  

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2080– 2094, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. naacl-main.168. URL https://aclanthology.org/2021.naacl-main.168.  

Si Qingyi, Wang Tong, Gu Naibin, Liu Rui, and Lin Zheng. Alpaca-cot: An instruction-tuning platform with unifed interface of instruction collection, parameter-effcient methods, and large language models. https://github.com/PhoebusSi/alpaca-CoT, 2023.  

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’20, pp. 3505–3506, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450379984. doi: 10.1145/3394486.3406703. URL https://doi.org/10. 1145/3394486.3406703.  

David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. Analysing mathematical reasoning abilities of neural models, 2019.  

Kumar Shridhar, Alessandro Stolfo, and Mrinmaya Sachan. Distilling reasoning capabilities into smaller language models. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 7059–7073, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.findings-acl.441.  

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. Preference ranking optimization for human alignment. arXiv preprint arXiv:2306.17492, 2023.  

InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https://github.com/InternLM/InternLM, 2023.  

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothe´e Lacroix, Baptiste Rozi\`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and effcient foundation language models, 2023a.  

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fne-tuned chat models, 2023b.  

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback, 2022.  

Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.  

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id $\equiv$ 1PL1NIMMrw.  

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. ArXiv, abs/2109.01652, 2021. URL https://api.semanticscholar.org/ CorpusID:237416585.  

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed Huai hsin Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Trans. Mach. Learn. Res., 2022, 2022a. URL https://api.semanticscholar. org/CorpusID:249674500.  

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. ArXiv, abs/2201.11903, 2022b. URL https://api.semanticscholar.org/ CorpusID:246411621.  

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023.  

Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, and Songfang Huang. How well do large language models perform in arithmetic tasks? arXiv preprint arXiv:2304.02015, 2023a.  

Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback without tears, 2023b.  

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. STar: Bootstrapping reasoning with reasoning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id $=$ _3ELRdg2sgI.  

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414, 2022.  

Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, and Joseph E. Gonzalez. The wisdom of hindsight makes language models better instruction followers, 2023.  

Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, Yongfeng Huang, Ruyi Gan, Jiaxing Zhang, and Yujiu Yang. Solving math word problems via cooperative reasoning induced language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4471–4485, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.acl-long.245.  

# A DETAILED EXPERIMENT SETTING  

# A.1 SFT ON GSM8K  

We fne-tune GSM8K with 3 epochs and a batch size of 128 on NVIDIA A100 GPUs. We use 8 GPUs for 7B and 13B models, 16 GPUs for 33B models, and 32 GPUs for 65B and 70B models during fne-tuning. We use a peak learning rate of 2e-5 with a $3\%$ learning rate warmup. We evaluate the results on the fnal epoch. We use greedy decode to calculate maj1 $@1$ and decode with temperature 0.7 to calculate maj1 $@100$ .  

# A.2 SFT ON DOWNSAMPLED GSM8K  

We random downsample GSM8K dataset for fne-tuning. We fnd that using 3 epochs for little data will result in very poor results which are listed in Table 5. We search training epoch among $\left\{3,\frac{3}{d a t a f r a c t i o n}\right\}$ and evaluate the latest epoch. We report better test results among these two different epoch settings.  

# A.3 REJECTION SAMPLING FINE-TUNING ON GSM8K  

We use an SFT model $\pi$ to sample on training dataset for $k\,=\,100$ times with a temperature of 0.7. We extract the equation list in generated reasoning paths by fnding $<<e q u a t i o n>>$ frst, removing all white spaces, and joining the equation string list by a special symbol to a string (called  

it get equation in our algorithm) for deduplication. We select the reasoning paths by this algorithm:   


<html><body><table><tr><td colspan="2">Algorithm 1: Reasoning Path Selection</td></tr><tr><td colspan="2">Data: Reasoning paths for question q, Rq Result: Selected reasoning paths for question q, Rq</td></tr><tr><td colspan="2"></td></tr><tr><td colspan="2"> Initialize selected reasoning paths, Rq = list()</td></tr><tr><td colspan="2"> Initialize appeared equation set, &g = set()</td></tr><tr><td colspan="2">for r in Rq do</td></tr><tr><td colspan="2"></td></tr><tr><td></td><td>if get-equation(r)  Eg then Rq.append(r);</td></tr><tr><td>6</td><td>Eq update([get-equation(r)])</td></tr><tr><td></td><td>end</td></tr><tr><td colspan="2">8 else</td></tr><tr><td>9</td><td>find rs E Rq s.t. get-equation(r?) = get-equation(r);</td></tr><tr><td>10</td><td></td></tr><tr><td>11</td><td>rs=r; end</td></tr><tr><td colspan="2">12 13 end 4 1end</td></tr></table></body></html>  

We are trying to fnd the most dissimilar reasoning paths based on Levenstein distances. The idea comes from we want diverse reasoning paths for better generalization.  

# B DETAILED RESULTS OF SFT AND RFT  

We list detailed results of SFT and RFT in Table 5 and 6.  

Table 5: Detailed numerical results in this paper, some experiments are still under running. We report maj1 $@1$ (accuracy) in this table.   


<html><body><table><tr><td>Model</td><td>Data</td><td>Epoch</td><td>7B</td><td>7B-2</td><td>13B</td><td>13B-2</td><td>33B</td><td>65B</td><td>70B-2</td></tr><tr><td>ICL-8shot</td><td>0</td><td>0</td><td>11.0</td><td>14.6</td><td>17.8</td><td>28.7</td><td>35.6</td><td>50.9</td><td>56.8</td></tr><tr><td>SFT</td><td>1/32</td><td>96</td><td>9.5</td><td>10.1</td><td>8.6</td><td>17.1</td><td>18.6</td><td>25.2</td><td>27.4</td></tr><tr><td>SFT</td><td>1/16</td><td>48</td><td>14.3</td><td>15.5</td><td>14.2</td><td>23.9</td><td>25.9</td><td>28.9</td><td>33.6</td></tr><tr><td>SFT</td><td>1/8</td><td>24</td><td>17.9</td><td>20.8</td><td>18.4</td><td>28.5</td><td>31.6</td><td>35.8</td><td>38.9</td></tr><tr><td>SFT</td><td>1/4</td><td>12</td><td>21.6</td><td>27.7</td><td>26.7</td><td>36.3</td><td>38.4</td><td>45.6</td><td>46.9</td></tr><tr><td>SFT</td><td>1/2</td><td>6</td><td>29.0</td><td>33.1</td><td>35.2</td><td>43.7</td><td>48.6</td><td>50.5</td><td>57.5</td></tr><tr><td>SFT</td><td>1/32</td><td>3</td><td>7.8</td><td>14.2</td><td>0.0</td><td>5.9</td><td>25.3</td><td>28.9</td><td>15.8</td></tr><tr><td>SFT</td><td>1/16</td><td>3</td><td>12.7</td><td>16.2</td><td>7.4</td><td>27.7</td><td>29.2</td><td>39.5</td><td>52.8</td></tr><tr><td>SFT</td><td>1/8</td><td>3</td><td>16.5</td><td>21.8</td><td>19.5</td><td>33.4</td><td>39.3</td><td>46.0</td><td>57.8</td></tr><tr><td>SFT</td><td>1/4</td><td>3</td><td>22.7</td><td>28.1</td><td>27.4</td><td>37.5</td><td>44.6</td><td>50.4</td><td>57.8</td></tr><tr><td>SFT</td><td>1/2</td><td>3</td><td>30.9</td><td>34.6</td><td>36.1</td><td>45.3</td><td>50.8</td><td>55.6</td><td>61.0</td></tr><tr><td>SFT</td><td>7.4K</td><td>3</td><td>35.9</td><td>41.6</td><td>43.0</td><td>50.0</td><td>54.6</td><td>59.3</td><td>63.2</td></tr><tr><td>RFT no dedup</td><td>1/32</td><td>3</td><td>37.5</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RFT no dedup</td><td>1/16</td><td>3</td><td>38.3</td><td>-</td><td></td><td>-</td><td></td><td></td><td></td></tr><tr><td>RFT no dedup</td><td>1/8</td><td>3</td><td>41.1</td><td>-</td><td></td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RFT no dedup</td><td>1/4</td><td>3</td><td>41.2</td><td>-</td><td></td><td></td><td>-</td><td></td><td>-</td></tr><tr><td>RFT no dedup</td><td>1/2</td><td>3</td><td>43.9</td><td>-</td><td></td><td>-</td><td></td><td></td><td>-</td></tr><tr><td>RFT no dedup</td><td>400K</td><td>3</td><td>43.6</td><td>46.7</td><td>46.9</td><td>53.7</td><td></td><td></td><td>-</td></tr><tr><td>RFT k=1</td><td>~12K</td><td>3</td><td>37.6</td><td>43.4</td><td>42.7</td><td>52.1</td><td>-</td><td>-</td><td></td></tr><tr><td>RFT k=3</td><td>～15K</td><td>3</td><td>39.0</td><td>45.3</td><td>45.2</td><td>51.9</td><td>-</td><td></td><td>-</td></tr><tr><td>RFT k=6</td><td>～18K</td><td>3</td><td>39.5</td><td>45.6</td><td>46.8</td><td>52.2</td><td></td><td>-</td><td>-</td></tr><tr><td>RFT k=12</td><td>~22K</td><td>3</td><td>41.6</td><td>45.3</td><td>48.0</td><td>53.1</td><td></td><td>-</td><td>-</td></tr><tr><td>RFT k=25</td><td>～28K</td><td>3</td><td>40.9</td><td>46.5</td><td>46.0</td><td>52.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RFT k=50</td><td>～35K</td><td>3</td><td>40.7</td><td>47.0</td><td>49.4</td><td>54.5</td><td></td><td></td><td>-</td></tr><tr><td>RFT k=100</td><td>~47K</td><td>3</td><td>41.7</td><td>47.5</td><td>49.1</td><td>54.8</td><td>54.5</td><td></td><td></td></tr><tr><td>RFT-U13B</td><td>104K</td><td>3</td><td>49.3</td><td>50.3</td><td>52.1</td><td>55.4</td><td>56.5</td><td>59.0</td><td>62.3</td></tr><tr><td>RFT-U33B</td><td>110K</td><td>3</td><td>49.1</td><td>51.2</td><td>51.4</td><td>55.3</td><td>57.9</td><td>59.7</td><td>64.8</td></tr></table></body></html>  

<html><body><table><tr><td>Setting</td><td>7B</td><td>7B-2</td><td>13B</td><td>13B-2</td><td>33B</td><td>65B</td><td>70B-2</td></tr><tr><td>ICL-8shot</td><td>11.0/18.1</td><td>14.6/-</td><td>17.8/29.3</td><td>28.7/-</td><td>35.6/53.1</td><td>50.9/69.7</td><td>56.8/-</td></tr><tr><td>SFT</td><td>35.9/48.7</td><td>41.6/55.4</td><td>43.0/55.2</td><td>50.0/61.7</td><td>54.6/72.6</td><td>59.3/69.7</td><td>63.2/73.5</td></tr><tr><td>RFTk=100</td><td>41.7/52.7</td><td>47.5/58.7</td><td>49.1/59.9</td><td>54.8/65.4</td><td>54.5/-</td><td></td><td></td></tr><tr><td>RFT-U13B</td><td>49.3/61.8</td><td>50.3/65.6</td><td>52.1/66.2</td><td>55.4/69.1</td><td>56.5/-</td><td>59.0/-</td><td>62.3/-</td></tr><tr><td>RFT-U33B</td><td>49.1/61.6</td><td>51.2/64.1</td><td>51.4/66.3</td><td>55.3/69.1</td><td>57.9/71.3</td><td>59.7/-</td><td>64.8/-</td></tr></table></body></html>  

Table 6: Detailed numerical results in this paper, some experiments are still under running. We report maj1 $@100$ in this table.   
Table 7: Cases of generated reasoning paths with different reasoning complexity from rejection sampling for RFT. The calculations are highlighted in red.   


<html><body><table><tr><td>QueryA</td><td>Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?</td></tr><tr><td>Path 1</td><td>Weng earned 12/60 = $<(12/60=0.2>>0.2 an hour for the 50 minutes that she worked yes- terday. Therefore, she earned $0.2 x 50 = $<<0.2*50=10>>10 for the 50 minutes that she</td></tr><tr><td>Path 2</td><td>workedyesterday.####10 = 70 * 0    d 7o=1>> = 0 / 71 s $<(50*0.2=10)>10 for the 50 minutes of babysitting. #### 10</td></tr><tr><td>Path 3</td><td>Since there are 60 minutes in an hour, then Weng did 50/60 = ((50/60=0.8333)>0.8333 of</td></tr><tr><td>Path 4</td><td>an hour of babysitting. So, she earned $12 x 0.8333 = $<<12*0.8333=10>>10. #### 10 She earns 12 *50 / 60= $<<12*50/60=10>>10. #### 10</td></tr><tr><td>Path 5</td><td>Weng earns 12 * (50 / 60)= $<<12*(50/60)=10>>10. #### 10</td></tr><tr><td>Query B</td><td>A local college is offeringGermanlessons and currentlyhas8 students enrolled.Through advertising, 8 more became interested but a fourth of these dropped out within a day. 2 more got frustrated and left. The class then rallied to show how simple the course actually is and increased enrollment by 5 times the amount of students already enrolled in the class, but 2 had to drop it because of scheduling conflicts. After one last rally, 6 more people enrolled. As the days passed, half of the class eventually dropped, and half of the</td></tr><tr><td>Path 1</td><td>remaining students graduated.How many are still enrolled? 8+8=<<8+8=16)>16 students are enrolled in the German class.8*.25=<<8*.25=2>>2 dropped out from the new recruits. 2 more left, so the class had 16-2-2=<<16-2-2=12>>12 students left. There were 12*5+12=<<12*5+12=72>>72 total students in the class after the rally. Two people had to drop the class, so there were 72-2=<<72-2=70>>70 left. 70+6=<<70+6=76>>76 were enrolled after the final rally. 76/2=<<76/2=38>>38 are left after half the class dropped. 38/2=<<38/2=19))19 are left after half the class graduated. #### 19</td></tr><tr><td>Path 2</td><td>8+8=<<8+8=16>>16 students are enrolled after the first rally. 16/4=<16/4=4>>4 dropped out. 16-4=<(16-4=12>>12 remain. 12*5+12=<<12*5+12=72>>72 after the second rally. 72-2=<<72-2=70>>70 after 2 more dropped. 70+6=<<70+6=76>>76 after 6 more enrolled. 76/2=<<76/2=38>>38 remain after half the class dropped. 38/2=<<38/2=19)>19 remain after half the class graduated. ####19</td></tr><tr><td>Path 3</td><td>Advertising brought in 8+8=<<8+8=16>>16 students. Of these, 16/4=<<16/4=4>>4 dropped out. 16-4-2=<(16-4-2=10>>10 remained. Then 10*5=<<10*5=50>>50 more enrolled. This brought the class to 50+10=<<50+10=60>>60 students. Then 60+6=<(60+6=66>>66 enrolled. So there were 66+10=<(66+10=76)>76 students. Then 76/2=<<76/2=38>>38 dropped. So 76-38=<<76-38=38>>38 remained. Then 38/2=<<38/2=19)>19 graduated. So 38-19=<<38-19=19>>19wereleft.####19</td></tr></table></body></html>  

# C CASE STUDY OF RFT  

In this section, we present the cases of the training samples from rejection sampling. The case studies would shed light on how RFT potentially improves the mathematical reasoning performance of LLMs. The cases are shown in Table 7. As aforementioned, RFT considers the reasoning paths with different calculation processes regarding equation forms or orders, leading to the correct answers. In the cases from Table 7, all the reasoning paths from RFT result in the correct answer of 10, while the calculation processes of reasoning are diverse. Path 1 and 2, as well as Path 4 and 5, are different in the equation forms as highlighted in red. Path 1 and 2 present a two-step calculation reasoning process while Path 4 and 5 alter to a one-step calculation reasoning process. The case demonstrates that rejection sampling can potentially provide more supervision signals that improve mathematical reasoning performance. The fltered reasoning paths sampled from LLMs themselves are of similar quality to the reasoning demonstrations from human annotations.  

# D PRELIMINARY EXPERIMENTS  

# D.1 SELF QUERY AUGMENTATION  

Through our preliminary experiments and case studies, the errors made by the fne-tuned LLMs are partly attributed to the incorrect reasoning chains where LLMs mistakenly understand the context information or fail to consider all the information in the queries. Although such incorrect reasoning chains lead to wrong answers to the original queries, the reasoning chains themselves represent reasonable logic. For example, for the query Josh decides to try fipping a house. He buys a house for $\it{\Omega}\times80,000$ and then puts in $\Delta50{,}O O O$ in repairs. This increased the value of the house by $I50\%$ . How much proft did he make?, a fne-tuned LLaMA model predicts The value of the house increased by $80,000*.I5{=}8I2,000$ . So the house was worth $80,O O O+I2,O O O{=}892,O O O$ . So he made a proft of $92,O O O-8O,O O O-5O,O O O=842,O O O$ where the model erroneously interprets $I50\%$ as $I5\%$ , but the reasoning chain is reasonable if we ignore the error.  

Therefore, such wrong predictions made by the LLMs may be correct under other queries (if we change $I50\%$ to $I5\%$ in the above example). We conduct experiments to generate queries for the predicted reasoning chains. This is a similar idea to the hindsight experience replay (Andrychowicz et al., 2017) in reinforcement learning where the method is designed to deal with the sparse reward problems by changing the original objectives for the failed samples to form samples with positive rewards. Such an idea was recently adopted by HIR (Zhang et al., 2023) to better align LLMs with instructions.  

Concretely, we reformat GSM8K reversely by predicting the query given the corresponding groundtrue reasoning result and then we fne-tune a LLaMA model on the reversed task. We use this model to generate queries on the predicted reasoning chains by a normally fne-tuned LLaMA model on the training set of GSM8K, formalizing a training sample for augmentation. We experiment on the LLaMA 7B model and fne-tune models on the data mixing original and generated samples or solely on generated samples.  

The results are shown in the left subfgure in Figure 8. We can see that fne-tuning with self query augmentation data leads to the worst results, and the performance of mixing the original data with self query augmented data still falls short of that of the original data. The fne-tuned performance for mathematical reasoning does not beneft from the naive idea of self query augmentation. Through several case studies of generated data, we fnd that there are two major defects in the generated data. The frst one is some reasoning chains themselves are not logically reasonable, for example, there may be some calculation errors in the reasoning chains. The second one is that the generated query may not be suitable for a reasoning chain. The query generation model may still erroneously interpret the information in the reasoning chains. Both defects attribute to a mediocre augmented data quality, hence can be possible reasons for the failure of this data augmentation procedure.  

# D.2 SELF REVISING AUGMENTATION  

We also explore improving the mathematical reasoning abilities of LLMs through revising augmentation. To equip LLaMA with revising abilities, we generate a revising dataset by frst sampling $K$ reasoning paths from a fne-tuned LLaMA model, then concatenating the query with one of the sampled reasoning paths using a template, and fnally pairing with the ground-true reasoning path to form a training sample. We use a sampling temperature of 0.7 for generating reasoning paths. During inference, we use the fne-tuned revising model to revise the prediction from the normally fne-tuned model.  

The results are shown in the middle subfgure of Figure 8. We can see that with $K=1$ the revising model improves the fnal accuracy marginally comparing $36.09\%$ to $35.90\%$ . Surprisingly, as we increase $K$ , the performances degrade. The possible defect of the revising model is that generated samples on the training set for revising training suffer from a distribution discrepancy with generated samples on the test set for revising inference. The sampled reasoning paths on the training set may  

![](https://cdn-mineru.openxlab.org.cn/extract/8432c534-d1a8-4d5b-b8b0-0fefd429f45a/5fa645e770322619aa1fe0dab5446f1c577a4efa12c0b92322ee65b4b00aa9cc.jpg)  
Figure 8: Results for different methods of self data augmentation. GSM. and H. represent GSM8K and Hindsight respectively. The red dotted lines in the middle and right fgures represent the results of vanilla fne-tuning on GSM8K.  

have a larger lexical similarity to the ground true reasoning paths compared to those on the test set.   
Therefore we try two different procedures to alleviate such an issue.  

1. We use the sampled reasoning path with the largest Levenstein distance out of $K$ sampled paths with respect to the ground true path to form a training sample.  

2. We split the train set to $N$ folds, and fne-tune a model on each $N-1$ folds and sampling reasoning path on the left fold.  

The results are shown in the middle and right subfgures in Figure 8, we can see that when leveraging Levenstein distance for reasoning path selection, the fne-tuned revising model enjoys a performance boost, harvesting uniformly better performance than the fne-tuning baseline across different $K$ ’s. The results demonstrate that for the revising performance, the lexical diversity of reasoning paths matters when constructing training samples. However, the revising performance does not beneft from the $N$ -fold procedure.  

# E ESTIMATING FLOPS OF SFT AND RFT  

We mainly follow the notations of (Kaplan et al., 2020) here.  

Training FLOPs For each input sample of length $n_{c t x}$ in GSM8K dataset, we can split it into two parts:  

$$
n_{c t x}=n_{Q}+n_{R}
$$  

where $n_{Q},n_{R}$ denotes the length of question and generated reasoning path and answers respectively.  

$$
C_{\mathrm{train}}\approx6N n_{c t x}N_{s}
$$  

where $N_{s}$ denotes the numbers of samples.  

Inference FLOPs We roughly computed the FLOPs of each token during the forward pass:  

$$
C_{\mathrm{forward}}(n_{\mathrm{ctx}})=2N+2n_{\mathrm{layer}}n_{\mathrm{ctx}}d_{\mathrm{model}}
$$  

To ensure the results were more accurate and reliable, we also took into account the Key-Value (KV) cache during the decoding procedure.  

$$
K V_{\mathrm{cache}}\approx4n_{\mathrm{layer}}d_{\mathrm{model}}^{2}
$$  

Therefore, we obtain the FLOPs per token during the forward pass considering the KV cache.  

$$
\begin{array}{r l}&{C_{\mathrm{forward}}^{'}(n_{c t x})=2N+2n_{\mathrm{layer}}n_{c t x}d_{\mathrm{model}}-K V_{\mathrm{cache}}}\\ &{\qquad\qquad=24n_{\mathrm{layer}}d_{\mathrm{model}}^{2}+2n_{\mathrm{layer}}n_{c t x}d_{\mathrm{model}}-4n_{\mathrm{layer}}d_{\mathrm{model}}^{2}}\\ &{\qquad\qquad=20n_{\mathrm{layer}}^{2}d_{\mathrm{model}}^{2}+2n_{\mathrm{layer}}n_{c t x}d_{\mathrm{model}}}\\ &{\qquad\qquad\approx1.66N+2n_{\mathrm{layer}}n_{c t x}d_{\mathrm{model}}}\end{array}
$$  

The total inference FLOPs are computed as follows:  

$$
C_{\mathrm{total}}=N_{s}\cdot[n_{q}C_{\mathrm{forward}}(n_{q})+\sum_{i=n_{q}}^{n_{q}+n_{r}}i\cdot C_{\mathrm{forward}}^{'}(i)]
$$  

where $N_{s}$ denotes the numbers of samples. $n_{q},n_{r}$ denotes the average length (tokens) of the user query and generated response respectively. In GSM8K dataset, $n_{q}\approx66$ and $n_{r}\approx130$ .  