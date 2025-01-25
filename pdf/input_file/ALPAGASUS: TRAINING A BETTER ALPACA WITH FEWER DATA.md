# ALPAGASUS: TRAINING A BETTER ALPACA WITH FEWER DATA  

Lichang Chen∗†, Shiyang Li ∗‡, $\mathbf{Jun}\;\mathbf{Yan}^{\sharp}$ , Hai Wang ‡, Kalpa Gunaratna‡, Vikas Yadav‡, Zheng Tang‡, Vijay Srinivasan‡, Tianyi Zhou†, Heng Huang†, Hongxia Jin‡  

† University of Maryland, College Park ‡ Samsung Research America ♯ University of Southern California   
{bobchen, tianyi, heng}@umd.edu   
{shiyang.li, h.wang2, k.gunaratna, vikas.y, zheng.tang,   
v.srinivasan, hongxia.jin}@samsung.com   
yanjun@usc.edu  

# ABSTRACT  

Large language models (LLMs) strengthen instruction-following capability through instruction-fnetuning (IFT) on supervised instruction/response data. However, widely used IFT datasets (e.g., ALPACA’s 52k data) surprisingly contain many lowquality instances with incorrect or irrelevant responses, which are misleading and detrimental to IFT. In this paper, we propose a simple and effective data selection strategy that automatically identifes and flters out low-quality data using a strong LLM (e.g., ChatGPT). To this end, we introduce ALPAGASUS, which is fnetuned on only 9k high-quality data fltered from the $52\mathrm{k}$ ALPACA data. ALPAGASUS signifcantly outperforms the original ALPACA as evaluated by GPT-4 on multiple test sets and the controlled human evaluation. Its 13B variant matches $>\,90\%$ performance of its teacher LLM (i.e., Text-Davinci-003 generating the 52k data) on test tasks. It also provides $5.7\mathrm{x}$ faster training, reducing the training time for a 7B variant from 80 minutes (for ALPACA) to 14 minutes 1. Moreover, the experiments prove the effcacy of our method across diverse datasets, base models, and LLM flters. Overall, ALPAGASUS demonstrates a novel data-centric IFT paradigm that can be generally applied to instruction-tuning data, leading to faster training and better instruction-following models. Our project page is available at: https://lichang-chen.github.io/AlpaGasus/.  

# INTRODUCTION  

Instruction fne-tuning (IFT) (Longpre et al., 2023) has been recently applied as an essential continual training stage for pre-trained large language models (LLMs) to achieve instruction-following capability (Ouyang et al., 2022b; Chen et al., 2023b), which is often attributed to aligning the models’ behavior with a diverse set of human instructions and responses (Taori et al., 2023; Askell et al., 2021). The recent series of open-sourced instruction-tuned models (Taori et al., 2023; Xu et al., 2023) reveal that the alignment of better IFT data could result in better instruction-following skills. For example, GPT-4-LLM (Peng et al., 2023) (with GPT-4 (OpenAI, 2023b) as its teacher) exhibits better reasoning and math ability than ALPACA (Taori et al., 2023) (with Text-davinci-003 as its teacher), though they share the same base model LLaMA (Touvron et al., 2023), demonstrating the importance of data quality.  

Although stronger teachers can usually bring further improvement by providing better IFT data, their responses inevitably include incorrect or irrelevant answers to the corresponding instructions (see examples in Fig. 2), which can be misleading or detrimental to IFT. Moreover, these data also increase unnecessary training costs. Alpaca-cleaned is the pioneer of fltering bad data in ALPACA dataset though it requires humans fully involved in examining and fltering the data. Nonetheless, how to automatically flter out poor-quality data from IFT datasets has not been investigated yet. A primary bottleneck is that rating the data quality usually requires expensive human labor but still may not be accurate for IFT because stronger teachers are more powerful in generating eloquent but incorrect responses that are more subtle to detect by humans. When considering datasets crafted by humans, such as the Dolly dataset (Dolly, 2023), assessing quality becomes even more intricate, given that responses stem from seasoned writers.  

This paper aims to bridge the gap by proposing a novel data-fltering strategy for IFT that is effcient, automatic, and accurate. Specifcally, we design a prompt applied to a powerful LLM (e.g., ChatGPT) for evaluating the quality of each (instruction, input, response) tuple and then flter out the ones with scores lower than a threshold. By applying this flter to the 52k data used to train ALPACA, we fnd that a majority of the data suffer from low-quality issues. Using the LLM flter, IFT on a much smaller but carefully fltered subset of 9k data produces a much better model, i.e., ALPAGASUS, than the original ALPACA, as shown in Fig. 1, following exactly the same training confguration of ALPACA. This also reduces the training time from 80 minutes to merely 14 minutes on $4\times$ NVIDIA A100 (80GB) GPUs. Moreover, we validate the versatility of our method, demonstrating its effectiveness on a range of datasets(e.g., Dolly, Alpaca, GPT4LLM), base models(e.g., LLaMA-1 and LLaMA-2), and LLM flters(e.g., ChatGPT and Claude-2). This discovery is inspiring, as it shows that the data quality in IFT can outweigh the quantity. In addition, this shift towards prioritizing data quality presents a new and more effcient paradigm that can generally improve the fne-tuning of LLMs.  

Our experiments include comprehensive evaluations for our ALPAGASUS, incorporating free-form instruction evaluation, various benchmarks, and human studies. We select four different human-instruction test sets for evaluating instruction-following capability, including the ones used by WizardLM (Xu et al., 2023), Vicuna (Chiang et al., 2023), Koala (Geng et al., 2023), and Self-Instruct (Wang et al., 2022). Given the notable advantages that GPT4 judge could match with both the controlled and crowdsourced human preferences $(>80\%$ agreement) (Zheng et al., 2023), we employ GPT-4 as our judge for the major evaluations. In the 7B and 13B model comparisons, ALPAGASUS performs signifcantly better than ALPACA on all four test sets. To address potential concerns regarding biases in model-based evaluations, we conduct human studies and benchmark evaluations, both of which corroborate the su  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/be309a54d3d5e9834c72a7a8def7deb87aa9df8e2027b2a9109ccfc388d45aab.jpg)  
Figure 1: Performance of ALPAGASUS on four test sets when increasing its fnetuning data, where the winning score is #Win−#Lose + 1 with #Testset $=\#\mathbf{W}\mathrm{in}+\#\mathrm{Tie}\;+$ #Lose to be the test set size and #Win/#Tie/#Lose to be the number of samples on which ALPAGASUS wins/ties/loses compared to ALPACA 52K.  

periority of our model compared to baseline counterparts. Furthermore, we present a fne-grained evaluation of ALPAGASUS on individual tasks including Generic, Roleplay, Knowledge, and Commonsense from the Vicuna test set. The results indicate ALPAGASUS exhibits advantages on a majority of the tasks.  

To sum up, our data-fltering approach exhibits signifcant benefts in terms of scalability and automation. We also demonstrate that prudent management of training data quality can lead to substantial performance improvement and computation savings of IFT. In addition, our data selection and evaluation strategies can generalize to other instruction fnetuning datasets and LLMs, thereby paving the way for a promising new research trajectory aimed at pragmatic LLM deployment.  

# 2 METHODOLOGY  

# 2.1 OVERVIEW  

Unlike the recent work (Zhou et al., 2023), which relies on human labor to curate 1k high-quality instruction data that leads to a better fnetuned model, we aim to avoid the expensive and timeconsuming human annotations. Hence, we exploit the potential of strong LLMs to be auto-graders of the training data and then flter out the data with lower scores.  

In particular, we prompt a strong API LLM, i.e., ChatGPT, to produce a score for each triplet of (instruction, input, response). The prompt is given in Fig. 3, where “dimension” denotes a  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/bdf1117629ce53d655aee914b38d8310bfcd2078ed063942be41027a4fb9116a.jpg)  
Figure 2: The fne-tuning pipeline of ALPAGASUS. We prompt ChatGPT as our auto-grader to score each training triplet on a scale of 0 to 5. We then use the exact same instruction fne-tuning script of ALPACA to train ALPAGASUS on the fltered data with scores higher than a threshold.  

# System Prompt:  

We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following.  

Instruction: [Instruction] Input: [Input] Response: [Response]  

# User Prompt:  

Please rate according to the [dimension] of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the [dimension]. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.  

Figure 3: Prompt $p_{G}$ to ChatGPT for rating and fltering training data in Eq. (1).  

user-preferred property such as helpfulness and accuracy. We then only select the triplets with scores higher than a certain threshold to fne-tune a LLaMA-series model following an existing IFT pipeline. Fig. 2 illustrates the data selection and training pipeline.  

# 2.2 DATA RATING AND FILTERING  

Given an IFT dataset $V$ of triplets $x=$ (instruction, input, response) with $x\in V$ and an open-sourced LLM $\theta$ (e.g., LLaMA), let $\theta_{V}$ denote the fnetuned $\theta$ on $V$ , our overarching goal is to select a subset $S\subset V$ such that IFT on $S$ results in a better model $\theta_{S}$ than $\theta_{V}$ .  

In order to select $S$ from $V$ , we prompt an API LLM $G(\cdot)$ (e.g., ChatGPT3) as an auto-grader rating each sample $x\in V$ by a score $G(x,p_{G})$ wherein $p_{G}$ is the rating prompt in Fig. 3. We then select $x_{i}$ whose score is above a certain threshold $\tau$ , i.e.,  

$$
S\triangleq\{x\in V:G(x,p_{G})\geq\tau\}.
$$  

We achieve $\theta_{S}$ by fnetuning $\theta$ on $S$ using an existing IFT framework.  

3We also use claude-2 as our response quality evaluator, which can be found in Appendix A.2  

# 2.3 ALPAGASUS: 9K TRAINING DATA FILTERED FROM ALPACA  

For “dimension” in the rating prompt $p_{G}$ shown in Fig. 3, given that “accuracy” closely aligns with human expectations of LLMs’ responses, we designate “accuracy” as the dimension for rating purposes.4 Correspondingly, we establish $\tau$ in Eq. (1) as an accuracy threshold for the subsequent experiments. The distribution of scores in relation to the 52k Alpaca dataset is presented in Fig. 4.  

In particular, we choose the threshold $\tau\,=\,4.5$ according to the score histogram. For the ALPACA dataset $V$ with 52,002 samples, this fltering criterion leads to a subset $S$ of 9,229 samples 5.  

# 3 EXPERIMENTAL SETUP  

# 3.1 FREE-FORM INSTRUCTION EVALUATION  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/8776ede99df455193a7d5c2945bea4110899af93162ba36eb59403a476bad342.jpg)  

Most instruction-tuned models are evaluated on one test set that might not cover suffcient diverse instructions and thus leads to a risk of biased evaluation (Chia et al., 2023). To conduct a holistic evaluation of ALPAGASUS, we curate our test sets from Self-instruct (Wang et al., 2022), Vicuna (Chiang et al.,  

Figure 4: Histogram of Scores (Alpaca Dataset).  

2023), WizardLM (Xu et al., 2023), and Koala (Geng et al., 2023), which together can cover more types of instructions and reduce the evaluation bias. Details of these four test sets are provided in Table 1.  

# 3.2 BASELINE MODELS  

We compare our ALPAGASUS with the following four recent LLMs.  

ALPACA (Taori et al., 2023) is an open-sourced model developed by Stanford University through IFT of LLaMA on a training dataset of 52,002 (instruction, input, response) samples with the responses generated by TextDavinci-003 (teacher).  

<html><body><table><tr><td>Test Set</td><td>#Samples</td><td>Category</td></tr><tr><td>Koala</td><td>180</td><td></td></tr><tr><td>Vicuna</td><td>80</td><td>√</td></tr><tr><td>WizardLM</td><td>218</td><td>√</td></tr><tr><td>Self-Instruct</td><td>252</td><td></td></tr></table></body></html>  

Table 1: Four test sets used in this paper.  

TEXT-DAVINCI-003 is an OpenAI LLM trained with an increased emphasis on contextual understanding and response accuracy. Its profciency in capturing complex linguistic patterns makes it a powerful teacher LLM for generating high-quality training data for fnetuning LLMs such as ALPACA.  

CHATGPT (OpenAI, 2023a) is an AI chatbot fnetuned via reinforcement learning with human feedback (RLHF). It exhibits exceptional capability across a wide range of tasks and might be the most popular chatbot recently. Hence, it would be interesting to study to what extent ALPAGASUS can match its performance.  

CLAUDE (Bai et al., 2022) is an AI chatbot developed by Anthropic. It was fnetuned by RLHF to align with humans’ preference on three dimensions, i.e., helpful, honest, and harmless. We use Claudev1.1 for comparison, which is comparable to ChatGPT on the AlpacaEval (Li et al., 2023).  

# 3.3 EVALUATION METRICS  

The evaluation of the instruction-following capability of LLMs is usually challenging due to the existence of multiple eligible responses to one instruction and the diffculty of reproducing human evaluations. In light of the recent advancements in automated evaluation (Dubois et al., 2023; Zheng et al., 2023; Chiang et al., 2023), which offer superior scalability and explainability than human studies, we also apply an API LLM $J(\cdot)$ (e.g., GPT-4) as the judge to evaluate $\theta_{S}$ and compare it with $\theta_{V}$ . In particular, we apply $J(\cdot)$ to compare the responses of $\theta_{S}$ and $\theta_{V}$ to each instruction $z$ drawn from a test set $D$ . Let $F(z;\theta_{V})$ and $F(z;\theta_{S})$ denote the two models’ responses to instruction $z\in D$ , the judge outputs a score for each response and we aim to achieve a higher score on $\theta_{S}$ , i.e.,  

$$
J(F(z;\theta_{S}))\ge J(F(z;\theta_{V}))
$$  

for most $z\in D$ . In our experiments, we include both models’ responses in the input to the judge (e.g., GPT-4), followed by an instruction to the judge, which aims to rate the responses with a score between 1 and 10. Details of the input and prompt to the judge can be found in Appendix $C^{6}$  

Since there exists position bias within LLM judges, which refers to a phenomenon where LLM judges have tendencies to prefer specifc positions over others (Wang et al., 2018; Ko et al., 2020; Wang et al., 2023), to mitigate it, we try both orders (i.e., placing ALPAGASUS’s response before/after the baseline model’s response) and defne the fnal judge of “Win-Tie-Lose” to be:(1) Win: ALPAGASUS wins twice, or wins once and draws once. (2) Tie: ALPAGASUS draws twice, or wins once and loses once. (3) Lose: ALPAGASUS loses twice, or loses once and draws once. To avoid cut-off responses, we allow models to generate up to 1024 tokens. For ChatGPT, Claude, and Text-Davinci-003, we set the temperature to 0.0, respectively, to reduce randomness and ensure a fair comparison.  

# 4 EXPERIMENTAL RESULTS  

# 4.1 QUALITY MATTERS MORE THAN QUANTITY  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/f9cc11a948e35a65efebeb586550f78a98c8ddf4936170220f9be83178845d51.jpg)  
Figure 5: Main results: comparing ALPAGASUS and ALPACA on their 7B and 13B models. ALPAGASUS-9k achieves much better performance than ALPACA-52k on all four test sets: Vicuna, Koala, Self-Instruct, and WizardLM.  

AlpaGasus- $\mathbf{\nabla}_{9\mathbf{k}}$ vs. Alpaca-52k We compare ALPAGASUS and ALPACA on two sizes of models in Fig. 5. They only differ in the training data: ALPACA uses all the 52k data while ALPAGASUS only uses 9k data selected from the 52k. Their hyperparameters and training scripts are the same. As shown in the evaluation results, ALPAGASUS signifcantly outperforms the original ALPACA across all four test sets. Moreover, when using LLaMA-2 as the base model, we observe consistent outcomes (See Appendix A.3). This consistency underscores the universality of our data fltering method, irrespective of the model choices. These fndings also confrm that our training data selection approach leads to superior performance even when the selected training data are only $17.75\%$ of the original dataset.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/98e2acca80efec859a44275f33391ea594bc18f4189e9098d866ade0922cea7c.jpg)  
Figure 6: Comparing ALPAGASUS with LLaMA fnetuned on randomly selected data.  

Quality-Guided Filtering vs. Random Filtering To investigate the effcacy of our data selection strategy, we compare ALPAGASUS with LLaMA models fne-tuned on a randomly sampled subset of the ALPACA $52\mathrm{k}$ data, denoted by ALPACA- $\mathbf{\nabla}\cdot9\mathbf{k}$ -random in Fig. 6. Both models start from the same initial model (i.e., LLaMA) and are then fnetuned on the same number of samples (i.e., 9k). They only differ in terms of the data selection criteria. In Fig. 6, we compare the two types of models under two model sizes, i.e., 7B and 13B. ALPAGASUS-9k signifcantly outperforms ALPACA-9k-random, showing the high quality of our selected data and their importance to the performance of IFT.  

# 4.2 HOW MUCH DATA SHOULD BE FILTERED?  

Threshold $\tau$ of data fltering. In Eq. (1), we select data with score $\geq\tau$ and we set $\tau=4.5$ in our main experiments, which results in 9k out of the 52k data to fnetune ALPAGASUS. To study the impact of the threshold $\tau$ on IFT, we compare ALPAGASUS with LLaMA fnetuned on $39\mathrm{k}$ data selected by applying a lower threshold of $\tau=4.0$ . We report the comparison results in Fig. 7. When tested on the Koala and WizardLM test sets, ALPACA-39k model outperforms the original ALPACA- $.52\mathrm{k}$ model. However, when using the Vicuna and Self-Instruct as test sets, ALPACA- $.39\mathrm{k}$ does not exhibit advantages over  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/8758c2bcae3af1ea740f15bd99933a98a1aa8154b82a96e8c8c1f8c068429f65.jpg)  

Figure 7: Comparing ALPACA-7B (39k data) with ALPACA-7B (52k data).  

the original ALPACA-52k model. Hence, a loose criterion (a lower threshold) includes more data in the selected data and a model with comparable performance as the original ALPACA. However, it still performs poorer than ALPAGASUS trained on much fewer but higher-quality data, indicating the negative impact of low-quality data to IFT.  

AlpaGasus trained on $3\mathbf{k}/6\mathbf{k}/9\mathbf{k}$ selected data. On the other hand, high-quality data show a positive impact on IFT. To verify this, we randomly draw 3k and 6k data from the 9k data selected for training ALPAGASUS and fnetune two variants of ALPAGASUS from LLaMA using the same training script. Fig. 8 reports the evaluation results of these variants: ALPAGASUS trained on 9k data performs the best on all four test sets, indicating that more high-quality data leads to better IFT models.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/d1c6ae493c9228bfc388a608c60a161a35b5a8aad03820bab09142719e688046.jpg)  

Figure 8: Comparing models fnetuned on $3\mathrm{k}/6\mathrm{k}/9\mathrm{k}$ high-quality data ( $3\mathbf{k}$ and 6k data are randomly drawn from the 9k data selected for ALPAGASUS).  

Minimum training data for AlpaGasus to match the performance of Alpaca. According to Fig. 1, ${\sim}6\mathrm{k}$ high-quality data suffces to fnetune LLaMA achieving similar performance as the original ALPACA.  

# 4.3 HUMAN STUDY  

We further undertake human studies by enlisting three participants tasked with labeling the question/answer pairs. To be specifc, we select 40 prompts from each test set, resulting in a total of 160 prompts. These are then presented to the participants alongside the corresponding responses generated by both ALPAGASUS-13B and Alpaca-13B. The fnal answers are determined by majority voting. There are 63/160 wins for ALPAGASUS-13B, 64/160 ties and 33/160 loses, which indicates the superiority of our ALPAGASUS. Comprehensive results on each test set and user guidelines could be found in Appendix J.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/cc00c292362d49235f84d2bd13fa018e9ec120d4558a219b0c63e7c2d2fa1c0c.jpg)  
Figure 9: ALPAGASUS-13B vs. Davinci-003, Claude, and ChatGPT. ALPAGASUS achieves average $90.1\%$ capacity of Davinci003, $81.2\%$ of Claude and $78.4\%$ of ChatGPT.  

# 4.4 COMPARISON WITH CHATGPT/CLAUDE/DAVINCI003.  

In Fig. 9, we compare ALPAGASUS with text-Davinci-003, ChatGPT, and Claude. The results show that ALPAGASUS-13B can achieve $\geq90\%$ capacity of its teacher model, text-Davinci-003, which is used to generate the ALPACA- $.52\mathrm{k}$ instruction data.  

# 4.5 BENCHMARK PERFORMANCE  

Following InstructEval (Chia et al., 2023), we also evaluate our models on benchmark datasets, i.e., MMLU (Hendrycks et al., 2020), DROP (Dua et al., 2019) Humaneval (Chen et al., 2021), BBH (Suzgun et al., 2022), to evaluate the models’ performance. The details of the benchmark setting can be found in Appendix B. Benchmark results of our ALPAGASUS are shown in Table 2, where higher values indicate better performance. ALPAGASUS-7B, 13B show superiority on the 3/4 datasets, which demonstrates the effectiveness of our fltering algorithm. Another interesting fnding is that the models trained with our fltered data can be better on all the benchmarks than training with randomly selected data.7  

<html><body><table><tr><td>Datasets</td><td>7B(9k-random)</td><td>7B(9k)</td><td>7B(52k)</td><td>13B(9k-random)</td><td>13B(9k)</td><td>13B(52k)</td></tr><tr><td>BBH</td><td>31.89</td><td>33.76</td><td>33.01</td><td>38.60</td><td>38.92</td><td>38.67</td></tr><tr><td>Drop</td><td>25.88</td><td>26.03</td><td>25.87</td><td>33.40</td><td>34.4</td><td>33.84</td></tr><tr><td>Humaneval</td><td>11.59</td><td>12.20</td><td>11.69</td><td>15.24</td><td>15.86</td><td>15.74</td></tr><tr><td>MMLU</td><td>36.93</td><td>38.78</td><td>40.86</td><td>44.98</td><td>46.12</td><td>47.89</td></tr></table></body></html>  

Table 2: The benchmark results of fltering the Alpaca dataset.  

# 5 HUMAN-WRITTEN INSTRUCTION SET FILTERING  

In addition to fltering machine-generated datasets, our approach is capable of fltering human-written datasets. Specifcally, we investigate the Databricks-dolly-15k dataset (Dolly, 2023), a seminal collection of 15,000 high-quality human-generated prompt/response pairs. Notably, this unparalleled dataset is a product of the collective efforts of more than 5,000 Databricks contributors and the included prompts and responses are more than just simple text; they embody a comprehensive spectrum of human cognition, covering activities from inventive brainstorming to succinct summarization.  

We also applied a threshold of 4.5 for data fltration, resulting in a fltered dataset of 2,996 samples. (Score distribution can be found in Appendix B) A comparison between the 7B/13B LLaMA trained on our fltered $3\mathbf{k}$ dataset and the one trained on the entire Dolly $15\mathrm{k}$ dataset is illustrated in Fig. 10 and Fig. 21. Our evaluation suggests that the model trained on our fltered data exhibits superior performance, thus underscoring the effcacy of our fltering method on human-composed datasets. Comprehensive details regarding training hyperparameters are provided in the Appendix D.8  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/801b441acc11aa280c6fac20c6099304ee9a679b2d22784e3e7e914dd5beb83b.jpg)  
Figure 10: Comparing models fnetuned on fltered 3k data and original Dolly $15\mathbf{k}$ data.  

# 6 CASE STUDY & ANALYSIS  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/55fb81ddf1273bc54c1d1ed19f99b5fc72f441f3a9f665518e83b859675cb2b2.jpg)  

Fig. 11 shows two case studies of 13B models trained on $52\mathrm{k}$ data (ALPACA), 9k selected data (ALPAGASUS), and 9k randomly selected data (ALPACA-9k-random). The left case study focuses on the math capability, where ALPAGASUS can produce a correct answer while ALPACA-9k-random cannot. As the judge, GPT-4 rates the answer of ALPAGASUS by a score of 10.0 while ALPACA$9\mathbf{k}$ -random receives a score of 2.0. The right case study focuses on coding skills, ALPACA-52k cannot follow the instructions but produces a regular expression to validate the website address while ALPAGASUS directly generates the correct code.  

We also conduct a fne-grained evaluation of ALPAGASUS on each skill/category in the WizardLM and Vicuna test sets, whose samples are split into a list of skill sets/categories and thus facilitate detailed analyses of the capabilities achieved by IFT (Appendix H). We compare two 7B models on the WizardLM test set and report the results in Fig. 25. Our ALPAGASUS achieves better or equally good performance than ALPACA on 22/29 skills but does not show advantages on the remaining 7 skills such as coding (e.g., code generation). To investigate the reasons, we notice that the coding categories include “python”, “Java”, $\mathrm{^{6}C++}^{\circ}$ , and “C#”, which indicate that we can allocate training samples regarding coding skills based on these related keywords (Appendix E). We fnd that our data selection/fltering, without specifying the proportions of skill categories, leads to a much higher fltering ratio of coding-related data $\begin{array}{r}{\frac{\widecheck{71}8-85^{\bullet}}{718}=88.16\%}\end{array}$ than the average fltering ratio 52002−9229 $\begin{array}{r}{\frac{52002-9229}{52002}\,=\,82.25\%}\end{array}$ . Hence, the resulting coding skill is weaker than other skills. This indicates the importance of keeping the training data diverse and balanced across different categories in IFT.  

# 7 COST SAVING  

We compare the training cost of ALPAGASUS and ALPACA in terms of the estimated expenses for the required computation on AWS. Notably, the training time is reduced from $80\mathrm{m}$ to $14\mathrm{m}$ for the 7B model and $5.5\mathrm{h}$ to 1h for the 13B model. Such training time reduction not only substantially enhances model iteration speed, but also reduces the cost from $\mathbb{S}27.31$ to $\mathbb{S}4.78$ for the 7B model and $\mathbb{S}225.28\$ to $\mathbb{S}40.96^{9}$ for the 13B model. It’s noteworthy that instruction-tuning 65B LLaMA models require a greater number of GPUs and an extended training duration. Consequently, as the model size scales up, our data selection method yields progressively pronounced cost savings.  

# 8 RELATED WORK  

Open-sourced Instruction-following models. Instruction-tuning datasets can be gathered in two ways. A number of studies (Köpf et al., 2023; Dolly, 2023; Zhou et al., 2023) utilize crowdsourcing to produce human-generated pairs of instructions and responses. This approach, while effective, can be laborious and costly. Alternatively, ALPACA (Taori et al., 2023) opens the door to create machine-generated IFT sets from the distillation of the “teacher” LLM, i.e., Text-Davinci-003. Peng et al. (2023) keep the instructions from ALPACA intact but using GPT-4 as the “teacher” LLM, which enhances model on 3H (Helpfulness, Honesty and Harmlessness) (Askell et al., 2021) alignment criteria. Vicuna (Chiang et al., 2023) is the frst to adopt ShareGPT (ShareGPT, 2023) data, which is the realistic dialogue data chatting with ChatGPT shared by users. Xu et al. (2023) and Luo et al. (2023) evolve the original Alpaca instruction set and obtain more complex instructions which help better elicit the instruction-following ability of LLMs. There also exists concurrent work like Koala (Geng et al., 2023) and UltraChat (Ding et al., 2023), using dialogue & preference data as well as the adversarial prompts to conduct safe alignment.  

Data-centric AI. Over the last decade, the realm of data-centric AI (Chu et al., 2016; Motamedi et al., 2021) has witnessed substantial progress. Central to this concept is the belief that the quality of data (Hajij et al., 2021; Zha et al., 2023; Chen et al., 2023a;c;d) warrants the same level of importance as algorithms within the AI/ML lifecycle. As noted by Chu et al. (2016), for an effective engagement with diverse types of data across various domains, data cleaning processes should exhibit a higher degree of automation and adaptability. With the advent of the Transformer architecture (Vaswani et al., 2017b), a shift in the paradigm of language models has occurred. Models such as RoBERTa (Liu et al., 2019), BERT (Vaswani et al., 2017a), and Bard 10 all have incorporated this effective structure, stacking varying quantities of transformer blocks to create more potent models. This marked a turning point in NLP research, signifying a heightened emphasis on data as opposed to model structure. Presently, SOTA LLMs like ChatGPT also underscore this shift toward data. They employ user data to conduct Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022a; Gao et al., 2022), which further aligns with the Data-centric AI philosophy.  

Evaluation of LLMs. Evaluating the open-ended instruction-following ability of LLMs is often neglected by previous works (Chung et al., 2022; Anil et al., 2023), though they conduct a series of benchmark evaluations centered around factuality (Hendrycks et al., 2020) and reasoning (Bisk et al., 2020) for their pre-training models. Similarly, the frameworks proposed by Liang et al. (2022) and Gao et al. (2021) focus more on the evaluation of the base models but not on the evaluation of the IFT models, where open-ended instruction-following capability are supposed to be prioritized. Since instruction-following is a general ability but the scope of benchmarks is limited, the recent works such as Koala (Geng et al., 2023), Vicuna (Chiang et al., 2023), Self-Instruct (Wang et al., 2022), and WizardLM (Xu et al., 2023) all provide the instruction sets they collected and some of them also include the categories of the instructions for the evaluation of instruction-tuned LLMs. There are also some leaderboards like Alpaca-Eval (Li et al., 2023) measuring the model’s instruction-following ability. Leveraging these recent advancements, we evaluate our models on human instruction sets.  

# 9 CONCLUSION  

In conclusion, our study reveals signifcant insights about the infuence of data quality over quantity in IFT. Through our proposed data-fltering method, we have demonstrated that relying on a small subset of high-quality IFT data can lead to LLMs that exhibit enhanced instruction-following capabilities, while also offering substantial computational advantages. Notably, our method proves versatile across different rating dimensions (e.g., Accuracy and helpfulness), LLM flters (e.g., ChatGPT and Claude-2), base model families (e.g., LLaMA-1 and LLaMA-2), model sizes (e.g., 7B and 13B), dataset types(e.g., machine-generated and human-written). By emphasizing the importance of data quality, we advocate for a transition in the existing paradigm where data accumulation has been a primary focus. This perspective transition can lead to more meaningful advancements in the feld of LLMs, making models more aligned with human intentions and less prone to errors induced by poor-quality data.  

# ACKNOWLEDGE  

Lichang Chen and Heng Huang were partially supported by U.S. NSF IIS 2347592, 2347604, 2348159, 2348169, DBI 2405416, CCF 2348306, CNS 2347617.  

# REFERENCES  

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.  

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.  

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.  

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artifcial intelligence, 2020.  

Jiuhai Chen, Lichang Chen, and Tianyi Zhou. It takes one to tango but more make trouble? in-context training with different number of demonstrations. arXiv preprint arXiv:2303.08119, 2023a.  

Lichang Chen, Jiuhai Chen, Tom Goldstein, Heng Huang, and Tianyi Zhou. Instructzero: Effcient instruction optimization for black-box large language models. arXiv preprint arXiv:2306.03082, 2023b.  

Lichang Chen, Minhao Cheng, and Heng Huang. Backdoor learning on sequence to sequence models. arXiv preprint arXiv:2305.02424, 2023c.  

Lichang Chen, Heng Huang, and Minhao Cheng. Ptp: Boosting stability and performance of prompt tuning with perturbation-based regularizer. arXiv preprint arXiv:2305.02423, 2023d.  

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.  

Yew Ken Chia, Pengfei Hong, Lidong Bing, and Soujanya Poria. Instructeval: Towards holistic evaluation of instruction-tuned large language models. arXiv preprint arXiv:2306.04757, 2023.  

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $90\%*$ chatgpt quality, March 2023. URL https: //lmsys.org/blog/2023-03-30-vicuna/.  

Xu Chu, Ihab F Ilyas, Sanjay Krishnan, and Jiannan Wang. Data cleaning: Overview and emerging challenges. In Proceedings of the 2016 international conference on management of data, pp. 2201–2206, 2016.  

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-fnetuned language models. arXiv preprint arXiv:2210.11416, 2022.  

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.  

Dolly. Free dolly: Introducing the world’s frst truly open instruction-tuned llm. Blog Post, 2023. URL https://www.databricks.com/blog/2023/04/12/ dolly-first-open-commercially-viable-instruction-tuned-llm.  

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proc. of NAACL, 2019.  

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. arXiv preprint arXiv:2305.14387, 2023.  

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPof, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021. URL https://doi.org/10.5281/zenodo.5371628.  

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. arXiv preprint arXiv:2210.10760, 2022.  

Xinyang Geng, Arnav Gudibande, Hao Liu, Eric Wallace, Pieter Abbeel, Sergey Levine, and Dawn Song. Koala: A dialogue model for academic research. Blog post, April 2023. URL https://bair.berkeley.edu/blog/2023/04/03/koala/.  

Mustafa Hajij, Ghada Zamzmi, Karthikeyan Natesan Ramamurthy, and Aldo Guzman Saenz. Datacentric ai requires rethinking data notion. arXiv preprint arXiv:2110.02491, 2021.  

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.  

Miyoung Ko, Jinhyuk Lee, Hyunjae Kim, Gangwoo Kim, and Jaewoo Kang. Look at the frst sentence: Position bias in question answering. arXiv preprint arXiv:2004.14602, 2020.  

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyf, et al. Openassistant conversations–democratizing large language model alignment. arXiv preprint arXiv:2304.07327, 2023.  

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval, 2023.  

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.  

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.  

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The fan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.  

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct, 2023.  

Mohammad Motamedi, Nikolay Sakharnykh, and Tim Kaldewey. A data-centric approach for training deep neural networks with less data. arXiv preprint arXiv:2110.03613, 2021.  

OpenAI. Chatgpt. https://openai.com/blog/chatgpt, 2023a.  

OpenAI. Gpt-4 technical report. arXiv, 2023b.  

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022a.  

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022b.  

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023.  

ShareGPT. Sharegpt. 2023. URL sharegpt.com.  

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.  

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.  

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and effcient foundation language models. arXiv preprint arXiv:2302.13971, 2023.  

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017a. URL https://proceedings.neurips.cc/paper_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.  

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017b.  

Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. arXiv preprint arXiv:2305.17926, 2023.  

Xuanhui Wang, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. Position bias estimation for unbiased learning to rank in personal search. In Proceedings of the eleventh ACM international conference on web search and data mining, pp. 610–618, 2018.  

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022.  

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions, 2023.  

Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, Zhimeng Jiang, Shaochen Zhong, and Xia Hu. Data-centric artifcial intelligence: A survey. arXiv preprint arXiv:2303.10158, 2023.  

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.  

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. arXiv preprint arXiv:2305.11206, 2023.  

# Appendix  

# Table of Contents  

A Frequently Asked Questions 14   
A.1 Is there any bias contained in the evaluation prompts? . . . 14   
A.2 Have you tried other LLM flter? . . 14   
A.3 What about the results on other base models, e.g., LLaMA-2? . . 15   
A.4 Can your LLM flter evaluate the stronger model’s responses, e.g., fltering the   
responses given by GPT-4? . . . 15   
A.5 Results on other rating dimensions, e.g., helpfulness? . . . . 16   
B Additional Results on Dolly Dataset 17   
B.1 Score Distribution . . . 17   
B.2 Benchmark results . . . . 17   
B.3 Dolly-13B Results . . . 18   
C Details of GPT-4 Evaluation Prompt 18   
D Training Hyperparameter Details 19   
D.1 Alpaca Dataset . . . . 19   
D.2 Dolly Dataset . . . 19   
E Keywords set for detailed analysis 19   
F Rated examples in Alpaca Dataset 20   
G Rated examples in Dolly Dataset 23   
H Analysis 26   
H.1 Analysis on WizardLM Test Set . . 26   
H.2 Analysis on Vicuna Test Set . . 27   
I Detailed Analysis on the WizardLM testset 27   
J Human Study 31   
K Limitations 31  

# A FREQUENTLY ASKED QUESTIONS  

A.1 IS THERE ANY BIAS CONTAINED IN THE EVALUATION PROMPTS  

We also explore alternate evaluation prompts such as the prompts provided by Zheng et al. (2023), which are shown in Table 3. We apply the same rules to calculate the “Win-Tie-Lose” and show the results in Fig. 12. Notably, ALPAGASUS consistently outperforms across all test sets.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/01fe963a6990c09cc1e19e34f39e2ad0ae89e0567deb4de9b4d12d1be7f4644e.jpg)  
Figure 12: The experimental results when using the evaluation prompt from Zheng et al. (2023) to judge the two responses. ALPAGASUS could still maintain its advantage.  

Table 3: The GPT-4 evaluation prompt from Zheng et al. (2023).   


<html><body><table><tr><td>System Prompt</td><td>Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict</td></tr><tr><td>Prompt Template</td><td>by strictly following this format: “[[A]]" if assistant A is better, “[[B]]" if assistant B is better, and “[[C]]" for a tie. [User Question] {question} [TheStartofAssistantA'sAnswer] {Answera} [TheEnd ofAssistantA'sAnswer] [TheStart ofAssistantB'sAnswer] {Answerb} [The End of Assistant B's Answer]</td></tr></table></body></html>  

# A.2 HAVE YOU TRIED OTHER LLM FILTER?  

Yes, we also try to use Claude- $\cdot2^{11}$ as our response quality evaluator (LLM flter). Fig. 13 and Fig. 14 demonstrate the score distribution and evaluation results on the four testsets, respectively. Remarkably, the 7B model instruction-tuned with $8\mathbf{k}$ selected data could be better than the model instruction-tuned with 52k Alpaca data on $3/4$ testsets and achieves signifcantly better over the model instruction-tuned with 8k random selected data.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/65538007446597b17f96e5e0890c9dab4f5913ff1e66830872ed57c23872fc0f.jpg)  
Figure 13: The score distribution of using Claude2 as the LLM flter.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/186e506df8bf0a88a4e02be73fd23db34eda392b8b9d50c2100e3920c437b6bd.jpg)  
Figure 14: The experimental results by using the Claude2 as response quality evaluator.  

As Fig. 13 shows, the interval between two scores is 1, which is different from the ChatGPT-based flter, where the interval is 0.5. Thus, if we would like to have fne-grained scores, a larger rating scale should be applied to the prompt as the present 5-point scale does not suffce. We leave the exploration of the rating scales to future work.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/d251b9974faffad6ee43447948576b453d73a91c871348c3e9a06a742f91aa72.jpg)  
A.3 WHAT ABOUT THE RESULTS ON OTHER BASE MODELS, E.G., LLAMA-2? We also have the results of LLaMA2 in Fig. 15, which shows the superiority of our method.   
Figure 15: The experimental results on LLaMA2. Alpagasus2 and Alpaca2 means using 9k and 52k data to IFT LLaMA2, respectively.  

# A.4 CAN YOUR LLM FILTER EVALUATE THE STRONGER MODEL’S RESPONSES, E.G., FILTERING THE RESPONSES GIVEN BY GPT-4?  

To answer the question, we apply our LLM flter to GPT4LLM (Peng et al., 2023) data. According to the score distribution, we use 4.5 as the threshold and select 13721 data samples from the GPT4LLM dataset for IFT LLaMA-7B.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/a0c08d5ea5bbcfe3b22cf9b816dc0f2a28dd90726f645ed8ccbfd0ec0dbe8f86.jpg)  
Figure 16: The score distribution of Alpaca-GPT4 dataset.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/b20b7204cb4fb76968400d278d1e65656a4019e4f76dabcf9629780c7f508fd0.jpg)  
Figure 17: The evaluation results on Alpaca-GPT4 dataset.  

The results presented in Fig. 17 demonstrate the superiority of our method on the Vicuna and WizardLM test sets. Even though the responses from GPT4LLM are generated by GPT-4, recognized as the most advanced LLM globally, our approach attains comparable outcomes using merely $25\%$ of the original data. Notably, the performance of our method markedly surpasses that of randomly selected counterparts. In summary, our LLM flter exhibits promise in discerning superior responses from teacher models.  

We also use “helpfulness” as our rating dimension and fnd that we only need $2\mathbf{k}$ data to train the base model that can surpass the base model trained with 52k Alpaca data. The score distributions are shown in Fig. 18.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/0178d11bbd69865aff1232daed975b2f25e8092d83f092d3c78ca51d3728bca0.jpg)  
Figure 18: The score distribution of helpfulness.  

Evaluation Results From Figure 19, it is evident that the models trained using our fltered Alpaca dataset outperform those trained on randomly selected datasets across all instruction test sets. Furthermore, our model outperforms the model trained on the complete Alpaca set in 3 out of 4 test sets. This underscores the signifcant potential of our fltering approach, especially considering that a model trained with a mere 2k data points can surpass one trained with the original $52\mathrm{k}$ Alpaca dataset.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/5f7e266fdda26de6a00cf6224b4f61f5d0f99c7548b5d2bc99ee289da7916dce.jpg)  
Figure 19: Evaluation results regarding on the “helpfulness” dimension.  

# B ADDITIONAL RESULTS ON DOLLY DATASET  

# B.1 SCORE DISTRIBUTION  

We show the score distribution of Dolly dataset(rated by ChatGPT) in Fig. 20.  

# B.2 BENCHMARK RESULTS  

We use the code provided by Chia et al. (2023) to conduct benchmark evaluation. For MMLU, BBH, Drop, and humaneval, we also use 5-shot, 3-shot, 3-shot, and 0-shot settings, respectively. We show the benchmark results in Table 4 of Dolly and the fltered set.  

Table 4: The benchmark results of fltering the Dolly dataset.   


<html><body><table><tr><td>Datasets</td><td>7B(3k-random)</td><td>7B(3k)</td><td>7B(15k)</td><td>13B(3k-random)</td><td>13B(3k)</td><td>13B(15k)</td></tr><tr><td>BBH</td><td>31.33</td><td>31.76</td><td>30.73</td><td>36.15</td><td>36.37</td><td>35.8</td></tr><tr><td>Drop</td><td>20.73</td><td>22.45</td><td>22.33</td><td>31.61</td><td>34.24</td><td>26.94</td></tr><tr><td>Humaneval</td><td>9.76</td><td>9.78</td><td>7.93</td><td>10.98</td><td>14.92</td><td>14.63</td></tr><tr><td>MMLU</td><td>35.01</td><td>35.83</td><td>36.25</td><td>44.39</td><td>46.92</td><td>46.13</td></tr></table></body></html>  

Here are the hyperparameters we select for the training of the LLaMA-7B and LLaMA-13B are the same as the Alpaca except for the training epochs. To avoid the under-train issue, we train 10 epochs,  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/a874f20ec127e870f7b14a7dede4bcb2297b1b10e86035efc3583ae00041c884.jpg)  
Figure 20: The score distribution of the Dolly.  

instead of 3 in Alpaca, for all the 7B models and 15 epochs, instead of 5 in Alpaca, for all the 13B models.  

# B.3 DOLLY-13B RESULTS  

We show the dolly-13B results. As Fig. 21 shows, our fltered Dolly dataset is better than the original Dolly dataset since it can achieve stronger instruction-following capacity of the instruction-tuned LLaMA-7B models via ours. (See the results on the four tests)  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/523e39c56d6b63a32cead78b59fd6130956481333661c2411c2e0f7e79b28cb1.jpg)  
Figure 21: Dolly 13B results. We show the dolly-13B results here. With the model size going up, our method can still perform pretty well.  

# C DETAILS OF GPT-4 EVALUATION PROMPT  

We provide the detailed form of the prompt to GPT-4 used for evaluation in Fig. 22. It is the prompt for evaluation used in the original Vicuna blog 12  

# System Prompt:  

You are a helpful and precise assistant for checking the quality of the answer.  

User Prompt:   
[Question]   
[The Start of Assistant 1's Answer] {answer_1}   
[The End of Assistant 1's Answer] [The Start of Assistant 2's Answer] {answer_2}   
[The End of Assistant 2's Answer]  

We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."  

Figure 22: The prompt for evaluation using GPT-4 as the judge.  

# D TRAINING HYPERPARAMETER DETAILS  

D.1 ALPACA DATASET We show the training hyperparameters and costs in Table 5. 13  

<html><body><table><tr><td>Model Size</td><td>Data Size</td><td>#GPUs</td><td>Epoch</td><td>LR</td><td>Batch Size</td><td>Time</td><td>Cost</td></tr><tr><td>7B</td><td>9k</td><td>4</td><td>3</td><td>2e-5</td><td>128</td><td>14m</td><td>$ 4.78*</td></tr><tr><td>7B</td><td>52k</td><td>4</td><td>3</td><td>2e-5</td><td>128</td><td>80m</td><td>$ 27.31*</td></tr><tr><td>13B</td><td>9k</td><td>8</td><td>5</td><td>1e-5</td><td>128</td><td>1h</td><td>$ 40.96</td></tr><tr><td>13B</td><td>52k</td><td>8</td><td>5</td><td>1e-5</td><td>128</td><td>5.5h</td><td>$ 225.28</td></tr></table></body></html>  

Table 5: All the cost is estimated based on the price provided by AWS. We assume the training scripts for all models are the same (e.g., training epochs, batch size on each GPU, accumulation steps, etc.)   
D.2 DOLLY DATASET We show the training hyperparameters in Table 6.   
Table 6: IFT hyperparameter details. (Dolly Dataset)   


<html><body><table><tr><td>Model Size</td><td>Data Size</td><td>Epoch</td><td>LR</td><td>Batch Size</td></tr><tr><td>7B</td><td>3k</td><td>3</td><td>2e-5</td><td>128</td></tr><tr><td>7B</td><td>15k</td><td>3</td><td>2e-5</td><td>128</td></tr><tr><td>13B</td><td>3k</td><td>5</td><td>1e-5</td><td>128</td></tr><tr><td>13B</td><td>15k</td><td>5</td><td>1e-5</td><td>128</td></tr></table></body></html>  

# E KEYWORDS SET FOR DETAILED ANALYSIS  

We use the keyword set of [Java, java, $C++$ , $\mathrm{c}++$ , C#, c#, Python, python] and count the number of (instruction, input, output) tuples which contain the keyword in this set.  

13https://aws.amazon.com/ec2/instance-types/p4/ a p4de.24xlarge(preview) node has 8 $\times\;80\mathrm{GB}$ A100 and it costs $\mathbb{S}40.96/\mathrm{h}$ .\*we assume training time of using 8 GPUs is half of using 4 GPUs  

F RATED EXAMPLES IN ALPACA DATASET  

We include more examples rated by the response quality evaluator, i.e., ChatGPT, in this section. The examples of Score 5.0, Score 4.5, Score 4.0, Score 3.5, Score 3.0, Score 2.5, Score 2.0 are shown in Table 7, Table 8, Table 9, and Table 10, respectively.  

[Instruction] Instruction: Suggest ways to reduce plastic waste in everyday life. [Input] None  

[Response] One way to reduce plastic waste in everyday life is to make the switch to reusable items. This includes bringing a reusable water bottle instead of buying bottled water, investing in a reusable coffee cup, bringing your own bags when shopping, and using reusable containers for storing and transporting food. You can also reduce the amount of plastic waste you create by avoiding or limiting the use of take-out or single-use containers, utensils, and straws. Additionally, try to fnd sustainable alternatives to plastic items when shopping and avoid purchasing items wrapped in plastic whenever possible. Finally, make sure you recycle any plastic items you do have.  

[Score & Explanation]: 4.5.The response provided by the AI assistant is highly accurate and relevant to the given instruction. It suggests various ways to reduce plastic waste in everyday life, including using reusable items, avoiding single-use containers and utensils, fnding sustainable alternatives, and recycling. The only potential improvement could be providing more specifc examples of sustainable alternatives to plastic items. Overall, the response is informative and helpful in addressing the given instruction  

[Instruction] Generate a list of non-alcoholic drinks.   
[Input] None   
[Response]   
- Fruit juices   
- Tea   
- Smoothies   
- Iced coffee   
- Milkshakes   
- Fizzy drinks   
- Hot chocolate   
- Lemonades   
- Kombucha   
- Energy drinks   
- Milkshake   
- Coconut water   
- Ginger beer   
- Ice tea   
- Sparkling water   
- Aloe vera drink   
- Vegetable juice   
[Score & Explanation]: 4.5. The AI assistant provided a comprehensive list of non-alcoholic drinks that are commonly consumed. The list includes a variety of options such as fruit juices, tea, smoothies, and fzzy drinks. However, there are some duplicates in the list such as milkshakes and iced coffee. Additionally, some of the options may not be as commonly consumed as others, such as kombucha and aloe vera drink. Overall, the response is accurate and provides a good range of options for non-alcoholic drinks.  

Table 8: The score $=4.5$ examples in Alpaca dataset.  

[Instruction] Instruction: Compare and contrast commercial banks and investment banks. [Input] None  

[Response] Commercial banks are fnancial institutions that provide services to their customers, such as loans, deposits, and investment management. Investment banks, on the other hand, are a type of fnancial institution that specializes in the trading of securities, such as stocks and bonds. The main difference between the two is that commercial banks provide services to customers, while investment banks provide services to governments, corporations, and wealthy individuals. Commercial banks are also subject to more regulations than investment banks, as they offer deposit accounts to customers. Investment banks, however, do not accept deposits from the general public and are subject to fewer regulations. Both types of banks are important for the global economy, but their roles and services are different.  

[Score & Explanation]: 4.0. The response provided by the AI assistant is mostly accurate and relevant to the given instruction and input. It clearly explains the differences between commercial banks and investment banks, highlighting their respective roles and services. However, it could have provided more specifc examples of the services offered by each type of bank, and could have elaborated further on the regulations that apply to them. Overall, the response is informative and helpful, but could beneft from more detail.  

[Instruction] Determine the application of statistics in the feld of economics [Input] None  

[Response] Statistics is widely used in economics to analyze and predict markets, examine the performance of companies, and evaluate economic policies. Statistics is used to identify trends and correlations and to measure variables such as prices and wages. Statistical models are also used to predict economic growth and changes in interest rates and to test the effects of regulation and taxation. Statistical tools are also used to estimate the impacts of policy changes, assess the effectiveness of government programs and identify the root causes of problems.  

[Score & Explanation]: 4.0. The AI assistant provided a fairly accurate response to the instruction. It correctly identifed the various applications of statistics in the feld of economics, including market analysis, company performance evaluation, policy evaluation, and prediction of economic growth and interest rates. However, the response could have been more detailed and provided specifc examples of statistical methods used in economics. Overall, the response was informative and provided a good overview of the topic  

Table 9: The triplets (instruction, input, response) obtain Score 4.0.  

G RATED EXAMPLES IN DOLLY DATASET [Instruction] Help me with spending a day in New Delhi. I like architecture, culture, and am a foodie.  

# [Context] None  

[Response] North India has a lot to offer for an aesthete, epicure, and a xenophile. Day 1 arrive in Delhi as many fights arrive early in the morning, settle into a hotel of your choice. Today travel around Delhi especially old Delhi and start your trip by visiting Red Fort which is an historical Mogul Fort commissioned by Mogul Emperor Shah Jahan in 1638. The fort is steeped in history and has a beautiful combination of Persian architecture with Indian infuence. After that have lunch in Chandi Chowk where you will fnd incredible street food and especially Karim’s which specializes in Mughlai food. Chandi Chowk is full of verve and energy and you will experience a slice of Indian culture. After the delicious lunch and walking through the narrow streets of Chandi Chowk, it is time to visit Qutb Minar a Unesco World Heritage Site which is considered the tallest minaret in the world at 240 ft constructed out of sandstone in 1192. Close by you can also visit Jamal Kalami mosque and relax at the nearby 200 acre park. In the evening you can travel to visit the Lotus Temple in Delhi which is a Bah˘00e1˘02bc˘00ed House of Worship and architected by Fariborz Sahba in 1976. For dinner, it is time to indulge yourself at Oberoi Delhi at Dhilli under the mentorship of Michelin-starred Chef Vineet Bhatia MBE and fnish your night at the rooftop bar Cirrus 9 with a smooth cocktail.  

[Score & Explanation]: 4.5. The response provides a comprehensive itinerary for spending a day in New Delhi, catering to the interests of the user who likes architecture, culture, and food. The assistant suggests visiting historical sites such as Red Fort and Qutb Minar, experiencing the culture and food of Chandi Chowk, and visiting the Lotus Temple. The assistant also recommends dining at a Michelin-starred restaurant and fnishing the night at a rooftop bar. The response is accurate and provides a well-rounded itinerary for a day in New Delhi. However, it could have been improved by providing more specifc details such as the opening and closing times of the suggested.  

# [Instruction] Why is the sky blue? [Context] None  

[Response] All the light in our daytime sky comes from the sun, and our sun produces light in many different colors. To understand this, just think of a prism. White light enters, and a rainbow of colors exits. Our sun is like that. Its light may seem uniform but it is actually emitting many different colors. Our sky though is not a mix of blue and red and green because the earth’s atmosphere is sandwiched between the sun and us, getting in the way. Atmospheric gases and particles scatter the incoming colors, bouncing them in all different directions. The amount of scattering increases as the wavelength of light decreases. Red light which has a long wavelength is scattered much less than blue light which has a short wavelength. As blue is scattered more, it flls our daytime sky. As the sun traverses the sky and moves low on the horizon, the light that reaches our eyes must actually travel further through the atmosphere. This has the effect of completely bouncing away all the blue light. We are left with the reds and yellows and oranges in wonderful mixture, making sunrises and sunsets so beautiful.  

[Score & Explanation]: 4.5. The response accurately explains why the sky appears blue during the day and why it appears red during sunrise and sunset. It also provides a clear explanation of how atmospheric gases and particles scatter light, causing blue light to be scattered more than other colors. The only potential improvement could be to provide a more concise explanation, as the response is somewhat lengthy. Overall, the response is highly accurate and informative.  

Table 12: The score $_{\cdot=4.5}$ examples in Dolly dataset.  

<html><body><table><tr><td>[Instruction] What is a bicameral legislature? [Context] None [Response] A legislative body comprised of two separate chambers. [Score & Explanation]: 4.0. The response provided by the AI assistant is accurate and directly answers the question asked in the instruction. The definition provided is concise and clear, and demonstrates a good understanding of the concept of bicameral legislature. However, the response could have been improved by providing additional information or examples to further clarify the concept.</td></tr><tr><td>[Instruction] When is tax day? [Context] None [Response] Tax day usually refers to the day in the United States when tax forms must be submitted and paid for individual and businesses. It is typically April 15 or later. [Score & Explanation]: 4.0. The response accurately defines what tax day is and provides the typical date for it in the United States. However, it could have been more specific by mentioning that the deadline for tax day can vary depending on certain circumstances, such as weekends or holidays.</td></tr><tr><td>[Instruction] Write a bullet point list of at least 7 crops that are grown in California. [Context] None [Response] - Artichokes -Almonds - Plums - Cherries - Garlic - Apples - Pears</td></tr><tr><td>[Score & Explanation]: 4.o. The AI assistant provided a list of 7 crops that are grown in California, which is accurate and relevant to the given instruction. However, there are many other crops that are also grown in California, and the assistant could have provided a more comprehensive list. Therefore, the response is not perfect but still satisfactory.</td></tr></table></body></html>  

Table 13: The score $=4.0$ examples in Dolly dataset.  

# [Context] None  

[Response] Among all ingredients, the most important element is cardamom. Of course, you cannot skimp on loose black tea leaves too!  

[Score & Explanation]: 2.5. The response mentions two important ingredients for making spiced chai - cardamom and loose black tea leaves. However, it does not provide any instructions on how to actually make the chai. The response is partially accurate as it addresses the context of making spiced chai but does not fully answer the instruction of providing the steps to make it.  

[Instruction] Is Seattle a nice place to live?  

[Context] Seattle came in at No. 39 in the 2023 World’s Best Cities Report, a comprehensive annual study by Resonance Consultancy. Seattle is a seaport city on the West Coast of the United States. The culture of Seattle is heavily defned by its signifcant musical history. Between 1918 and 1951, nearly 24 jazz nightclubs existed along Jackson Street, from the current Chinatown/International District to the Central District. The jazz scene nurtured the early careers of Ernestine Anderson, Ray Charles, Quincy Jones, and others. The city is also the origin of rock acts such as Foo Fighters, Heart, and Jimi Hendrix, as well as the grunge subgenre of rock and its pioneering bands Alice in Chains, Pearl Jam, Nirvana, and Soundgarden. The city itself is hilly, though not uniformly so. Like Rome, the city is said to lie on seven hills; the lists vary but typically include Capitol Hill, First Hill, West Seattle, Beacon Hill, Queen Anne, Magnolia, and the former Denny Hill. Due to its location in the Pacifc Ring of Fire, Seattle is in a major earthquake zone. Temperature extremes are moderated by the adjacent Puget Sound, greater Pacifc Ocean, and Lake Washington. Thus extreme heat waves are rare in the Seattle area, as are very cold temperatures (below about 15 F). The Seattle area is the cloudiest region of the United States, due in part to frequent storms and lows moving in from the adjacent Pacifc Ocean. With many more "rain days" than other major American cities, Seattle has a well-earned reputation for frequent rain. Seattle’s economy is driven by a mix of older industrial companies and "new economy" internet and technology companies, as well as service, design, and clean technology companies.  

# [Response] Yes.  

[Score & Explanation]: 2.5. The AI assistant’s response is not accurate enough. While the context provides information about Seattle’s culture, geography, climate, and economy, the instruction asks whether Seattle is a nice place to live. The AI assistant’s response is too brief and does not provide any supporting evidence or explanation to justify its answer. Therefore, the response is not helpful or informative enough for the user.  

[Instruction] Best workout in the world.  

# [Context] None  

[Response] Walk. Just walk as much as you can.  

[Score & Explanation]: 2.0. The response provided by the AI assistant is not very accurate. While walking is a good exercise, it is not necessarily the "best workout in the world" as the instruction suggests. The response also lacks specifcity and does not provide any additional information or guidance on how to make the most out of the workout. Therefore, I would rate the accuracy of the response as 2 out of 5.  

Table 14: The score $=\!2.0$ and score $=\!2.5$ examples in Dolly dataset.  

# H ANALYSIS  

# H.1 ANALYSIS ON WIZARDLM TEST SET  

We conduct a fne-grained evaluation of ALPAGASUS on each skill/category in the WizardLM and Vicuna test sets, whose samples are split into a list of skill sets/categories and thus facilitate detailed analyses of the capabilities achieved by IFT.  

ALPAGASUS-7B(9k) vs. ALPACA-7B(52k). We compare these two 7B models on the WizardLM test set and report the results in Fig. 25. Our ALPAGASUS achieves better or equally good performance than ALPACA on 22/29 skills but does not show advantages on the remaining 7 skills such as coding (e.g., code generation). To investigate the reasons, we notice that the coding categories include “python”, “Java”, $\mathrm{{}^{\bullet}C++}\mathrm{{}^{\bullet}}$ , and ${}^{\bullet}\mathbf{C}\#{}^{\bullet}$ , which indicate that we can allocate training samples regarding coding skills based on these related keywords (Appendix E). We fnd that our data selection/fltering, related data without specifying the proportions of skill categories, leads to a much higher fltering ratio of coding- $\begin{array}{r}{\frac{7\check{1}8-85}{718}=88.16\%}\end{array}$ than the average fltering ratio $\begin{array}{r}{\frac{52002-\mathcal{\widetilde{9}}229}{52002}=82.\dot{2}5\%}\end{array}$ . Hence, the resulting coding skill is weaker than other skills. This indicates the importance of keeping the training data diverse and balanced across different categories in IFT.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/3dc66b79e59d736718360ec2dd8b8d6e5f00a8b161fdf1947b1577923cbd455a.jpg)  
H.2 ANALYSIS ON VICUNA TEST SET   
Figure 23: Fine-grained evaluation of ALPAGASUS-13B-9k vs. ALPACA-13B-52k on categories of the Vicuna test set.  

Fig. 23 demonstrates the detailed analysis on Vicuna testset. ALPAGASUS-7B is better than the ALPACA-7B in the majority of the categories, including Counterfactual, Roleplay, Knowledge, and Generic, etc. Another strong point is that when the base model scales up, the conclusion still holds. (See right part of the Fig. 23)  

In Fig. 26, Fig. 27, and Fig. 28, we compare ALPAGASUS with text-Davinci-003, ChatGPT, and Claude, respectively. The results show that ALPAGASUS-13B can achieve $\ge91\%$ capacity of its “teacher” model, text-Davinci-003 (all the responses in the ALPACA-52k dataset are generated by text-Davinci-003 so we call it “teacher” LLM). The results also show that our model could achieve pretty good performance on tasks like Writing, RolePlay, Toxicity, Art, etc., while it still needs improvement on coding and math capacity when compared with stronger LLMs.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/4471212cd041574d6e9976fdea5b3c8978b87b4882ab773f18b7c9c128504cfc.jpg)  
Figure 24: Fine-grained evaluation of ALPAGASUS-9k(13B) vs. ALPACA- $.52\mathrm{k}$ (13B) on categories of the WizardLM test set.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/4985632b22a36433ca6de567b09cf55d0a780e815fcd6f22e45899ca2888f05b.jpg)  
Figure 25: Fine-grained evaluation of ALPAGASUS-9k(7B) vs. ALPACA-52k(7B) on categories of the WizardLM test set.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/46457d4da652217e4fe88057160ca0c25a5c9767272fd7dfa952e651d615c53f.jpg)  
Figure 26: Compare with ChatGPT. Achieve average $78.26\%$ capacity of ChatGPT on all 29 skills.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/61928fa4c41474ac49c2981b1010fb693a0c4892f9da0cb0a7981fc6d547c129.jpg)  
Figure 27: Compare with Claude-v1. Achieve average $78.41\%$ capacity of ChatGPT on all 29 skills.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/8928434d575704cbae3f56036849a631691aca2b5ab84e4d8d223f183badf15b.jpg)  
Figure 28: Compare with Davinci-003. Achieve an average $91.11\%$ capacity of ChatGPT on all 29 skills.  

# J HUMAN STUDY  

We conduct the human study among three different users. The evaluation interface is shown as Table 15:  

You’ll be presented with a series of questions. For each question, two answers   
will be provided. Your task is to read both answers carefully and decide which   
one you believe is better. When judging, consider:   
Relevance: Does the answer directly address the question?   
Completeness: Is the answer comprehensive?   
Coherence: Is the answer logically structured and easy to understand?   
Accuracy: Is the information provided in the answer correct?  

Question: <QUESTION>  

Answer A: Answer B: <ANSWER A> <ANSWER B>  

1. Answer A is signifcantly better.   
2. Answer B is signifcantly better.   
3. Neither is signifcantly better.  

Table 15: Human annotation interface.  

We show more detailed results of human evaluations in Fig. 29.  

![](https://cdn-mineru.openxlab.org.cn/extract/ef895a9b-5eee-422a-bda7-cd5692f6936d/7f5bab4185fbcf158b1cbd0b846e639de703b4b7ef6221832877dd00daf63280.jpg)  
Human Study:Alpagasus-13B(9k) vs. Alpaca-13B(52k)   
Figure 29: The detailed results of human study.  

# K LIMITATIONS  

Model Size. In our experiments, we evaluated our IFT strategy by training models of two different sizes, 7B and 13B, since they are the most common sizes for recent open-source LLMs. We plan to extend this study to larger model sizes such as 33B, 65B, or even 175B, and verify whether the same conclusion still holds, i.e., a small subset of high-quality data selected by our method can improve the instruction-fnetuned model. We leave analysis on the IFT of larger models as future work.  