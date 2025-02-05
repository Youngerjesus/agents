# Curating Trillion-Token Datasets: Introducing NVIDIA NeMo Data Curator

The [latest developments in large language model (LLM) scaling laws](https://arxiv.org/abs/2203.15556) have shown that when scaling the number of model parameters, the number of tokens used for training should be scaled at the same rate. The [Chinchilla](https://arxiv.org/abs/2203.15556) and [LLaMA](https://arxiv.org/abs/2302.13971) models have validated these empirically derived laws and suggest that previous state-of-the-art models have been under-trained regarding the total number of tokens used during pretraining.

Considering these recent developments, it’s apparent that LLMs need larger datasets, more than ever.

However, despite this need, most software and tools developed to create massive datasets for training LLMs are not publicly released or scalable. This requires LLM developers to build their own tools to curate large language datasets.    

To meet this growing need for large datasets, we have developed and released the NeMo Data Curator: a scalable data-curation tool that enables you to curate trillion-token multilingual datasets for pretraining LLMs.

Data Curator is a set of Python modules that use [Message-Passing Interface (MPI)](https://www.open-mpi.org/) and [Dask](https://www.dask.org/) to scale the following tasks to thousands of compute cores:

+   Data download
+   Text extraction
+   Text reformatting and cleaning
+   Quality filtering
+   Exact or fuzzy deduplication

Applying these modules to your datasets helps reduce the burden of combing through unstructured data sources. Through document-level deduplication, you can ensure that models are trained on unique documents, potentially leading to greatly reduced pretraining costs.

In this post, we provide an overview of each module in Data Curator and demonstrate that they offer linear scaling to more than 1000 CPU cores. To validate the data curated, we also show that using documents it processes from [Common Crawl](https://commoncrawl.org/) for pretraining provides significant downstream task improvement over using raw downloaded documents.

## Data-curation pipeline[](#data-curation_pipeline)

This tool enables you to download data and extract, clean, deduplicate, and filter documents at scale. Figure 1 shows a [typical LLM data-curation pipeline](https://arxiv.org/abs/2005.14165) that can be implemented. In the following sections, we briefly describe the implementation of each of these modules available.

![Workflow diagram depicts the download and extraction, fuzzy deduplication, and quality-filtering stages of a LLM data-curation pipeline. ](https://developer-blogs.nvidia.com/wp-content/uploads/2023/08/llm-data-curation-pipeline.png)

*Figure 1. A common LLM data-curation pipeline for datasets like Common Crawl that can be implemented with the modules available within the Data Curator*

### Download and text extraction[](#download_and_text_extraction)

The starting point for preparing custom pretraining datasets for many LLM practitioners is a list of URLs that point to data files or websites that contain content of interest for LLM pretraining.

Data Curator enables you to download pre-crawled web pages from data repositories such as [Common Crawl](https://commoncrawl.org/), [Wikidumps](https://dumps.wikimedia.org/), and [ArXiv](https://info.arxiv.org/help/bulk_data_s3.html) and to extract the relevant text to [JSONL](https://jsonlines.org/) files at scale. Data Curator also provides you with the flexibility of supplying your own download and extraction functions to process datasets from a variety of sources. Using a combination of MPI and Python Multiprocessing, thousands of asynchronous download and extraction workers can be launched at runtime across many compute nodes.

### Text reformatting and cleaning[](#text_reformatting_and_cleaning)

Upon downloading and extracting text from documents, a common step is to fix all Unicode-related errors that can be introduced when text data are not properly decoded during extraction. Data Curator uses the [Fixes Text For You library (ftfy)](https://github.com/rspeer/python-ftfy) to fix all Unicode-related errors. Cleaning also helps to normalize the text, which results in a higher recall when performing document deduplication.

### Document-level deduplication[](#document-level_deduplication)

When downloading data from massive web-crawl sources such as Common Crawl, it’s common to encounter both documents that are exact duplicates and documents with high similarity (that is, near duplicates). Pretraining LLMs with repeated documents can lead to [poor generalization](https://arxiv.org/abs/2205.10487) and a [lack of diversity during text generation](https://arxiv.org/abs/2107.06499).

We provide exact and fuzzy deduplication utilities to remove duplicates from text data. The exact deduplication utility computes a 128-bit hash of each document, groups documents by their hashes into buckets, selects one document per bucket, and removes the remaining exact duplicates within the bucket.

The fuzzy-deduplication utility uses a [MinHashLSH-based approach](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf) where MinHashes are computed for each document, and then documents are grouped using the locality-sensitive property of min-wise hashing. After documents are grouped into buckets, similarities are computed between documents within each bucket to check for potential false positives created during `MinHashLSH`. 

Initially, the deduplication phases of the Data Curator pipeline were implemented in CPU-only logic. Given the rapid evolution of LLM training techniques and growing training corpora, we found that we needed to repeatedly run the lengthy and expensive deduplication phase. We found that using the RAPIDS framework for GPU-based data processing, made it possible to accelerate the types of text comparisons and graph algorithms used in deduplication.

Using the popular 4.5-TB RedPajama dataset as an example, the initial CPU-based deduplication stage completed in 37 hours, using 20 high-end CPU nodes with 188 GB of RAM and 48 CPU cores per node. On four DGX A100 nodes (8x 80-GB GPUs each), deduplication now completes in 3 hours. That enables the pipeline to scale to multiple nodes and run much faster, making it possible to curate large-scale datasets for foundation models in hours instead of days.

With GPU acceleration, organizations can achieve 20x faster and 5x cheaper deduplication than the CPU version. This GPU-accelerated deduplication will be released in an upcoming version of NeMo Data Curator.

### Document-level quality filtering[](#document-level_quality_filtering)

In addition to containing a significant fraction of duplicate documents, data from web-crawl sources such as Common Crawl often tend to include many documents with informal prose. This includes, for example, many URLs, symbols, boilerplate content, ellipses, or repeating substrings. They can be considered low-quality content from a language-modeling perspective.

While it’s been shown that diverse LLM pretraining datasets lead to [improved downstream performance](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/), a significant quantity of low-quality documents can [hinder performance](https://arxiv.org/abs/2109.00698).  Data Curator provides you with a highly configurable document-filtering utility that enables you to apply custom heuristic filters at scale to your corpora. The tool also includes implementations of language-data filters (both classifier and heuristic-based) that have been shown to [improve overall data quality and downstream task performance](https://arxiv.org/abs/2112.11446) when applied to web-crawl data.

## Scaling to many compute cores[](#scaling_to_many_compute_cores)

To demonstrate the scaling capabilities of the different modules available within Data Curator, we used them to prepare a small dataset consisting of approximately 40B tokens. This involved running the previously described data-curation pipeline on 5 TB of Common Crawl WARC files.

For each pipeline stage, we fixed the input dataset size while linearly increasing the number of CPU cores used to scale the data curation modules (that is, strong scaling). We then measured the speedup for each module. The measured speedups for the quality-filtering and fuzzy-deduplication modules are shown in Figure 2. 

Examining the trends of the measurements, it’s apparent that these modules can reach substantial speedups when increasing the number of CPU cores used for distributing the data curation workloads. Compared to the linear reference (orange curve), we observe that both modules are able to achieve considerable speedup when using up to 1,000 CPUs or more.

![Chart shows compute-scaling curves on the speedup achieved when scaling the fuzzy-deduplication and quality-filtering modules of the NeMo Data Curator to many CPUs.](https://developer-blogs.nvidia.com/wp-content/uploads/2023/08/measured-speedup-data-curator.png)

*Figure 2. Measured speedup for the fuzzy-deduplication and quality-filtering modules within Data Curator*

## Curated pretraining data results in improved model downstream performance[](#curated_pretraining_data_results_in_improved_model_downstream_performance)

In addition to verifying the scaling of each module, we also performed an ablation study on the data curated from each step of the data-curation pipeline implemented within the tool. Starting from a downloaded Common Crawl snapshot, we trained a 357M parameter GPT model on 78M tokens curated from this snapshot after extraction, cleaning, deduplication, and filtering.

After each pretraining experiment, we evaluated the model across the RACE-High, PiQA, Winogrande, and Hellaswag tasks in a zero-shot setting. Figure 3 shows that the results of our ablation experiments averaged over all four tasks. As the data progresses through the different stages of the pipeline, the average over all four tasks increases significantly, indicating improved data quality.

![Bar graph shows the improvement in LLM downstream task performance when trained on cleaned, deduplicated, and filtered text.](https://developer-blogs.nvidia.com/wp-content/uploads/2023/08/dataset-ablation-results.png)

*Figure 3. Results of dataset ablation tests for a 357M parameter model trained on data generated from each stage of the processing pipeline within NeMo Data Curator*

## Curating a 2T token dataset with NeMo Data Curator[](#curating_a_2t_token_dataset_with_nemo_data_curator)

Recently, the [NVIDIA NeMo service](https://www.nvidia.com/en-us/gpu-cloud/nemo-llm-service/) started providing early-access users with the opportunity to customize an NVIDIA-trained 43B-parameter multilingual large foundation model. To pretrain this foundation model, we prepared a dataset consisting of 2T tokens that included 53 natural languages originating from a variety of diverse domains as well as 37 different programming languages.

Curating this large dataset required applying our data-curation pipeline implemented within Data Curator to a total of 8.7 TB of text data on a CPU cluster of more than 6K CPUs. Pretraining the 43B foundation model on 1.1T of these tokens resulted in a state-of-the-art LLM that’s currently being used by NVIDIA customers for their LLM needs.

## Conclusion[](#conclusion)

To meet the growing demands for curating pretraining datasets for LLMs, we have released Data Curator as part of the [NeMo framework](https://developer.nvidia.com/nemo). We have demonstrated that the tool curates high-quality data that leads to improved LLM downstream performance. Further, we have shown that each data-curation module available within Data Curator can scale to use thousands of CPU cores.

We anticipate that this tool will significantly benefit LLM developers attempting to build pretraining datasets.