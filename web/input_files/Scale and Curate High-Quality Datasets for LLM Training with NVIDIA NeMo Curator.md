# Scale and Curate High-Quality Datasets for LLM Training with NVIDIA NeMo Curator

Enterprises are using [large language models](https://www.nvidia.com/en-us/glossary/large-language-models/) (LLMs) as powerful tools to improve operational efficiency and drive innovation. [NVIDIA NeMo microservices](https://developer.nvidia.com/blog/simplify-custom-generative-ai-development-with-nvidia-nemo-microservices) aim to make building and deploying models more accessible to enterprises. An important step for building any LLM system is to curate the dataset of tokens to be used for training or customizing the model. 

However, curating a suitable dataset is a challenging task. Diversity, relevance, and quality of the data are all factors that affect the ability of the model to perform well. The data should also comply with data protection regulations and respect the privacy of individuals.

In this post, we discuss the open-source version of the [NVIDIA NeMo Curator framework](https://developer.nvidia.com/nemo-microservices-early-access), the foundation upon which the recently introduced NeMo Curator microservice is built. NeMo Curator aims to simplify and streamline the data curation process, paving the way for the adoption of generative AI at an enterprise scale.

## NeMo Curator simplifies and scales data curation pipelines[](#nemo_curator_simplifies_and_scales_data_curation_pipelines)

NeMo Curator supports data curation for model pretraining and was engineered on the following key pillars: performance, scalability, and customizability.

It can seamlessly scale across thousands of compute cores and uses highly optimized CUDA kernels to effortlessly perform a variety of data acquisition, preprocessing, and cleaning tasks, enabling enterprise developers to focus on problem-solving.

Built with extensibility and flexibility in mind, NeMo Curator enables developers to customize data curation pipelines to suit their business needs and address their unique challenges. Each component can be quickly customized via easy-to-use configuration files. 

Simultaneously, the Pythonic API of the framework offers deeper customization of the data curation pipeline with a few lines of code. 

Today, NeMo Curator provides the following functionality out of the box:

+   Data download and extraction
+   Text cleaning and language identification
+   Quality filtering
+   Privacy filtering
+   Domain and toxicity classification
+   Deduplication
+   Streamlined scalability
+   Support for model customization tasks

### Data download and extraction[](#data_download_and_extraction)

NeMo Curator comes with several helpers for downloading and extracting data from commonly used sources. 

Out of the box, NeMo Curator can download [Common Crawl](https://commoncrawl.org/get-started) snapshots, [arXiv bulk data from Amazon S3](https://info.arxiv.org/help/bulk_data_s3.html) and Wikipedia. It also provides helpers for text extraction and preparation for subsequent data operations by organizing the downloaded data into the [JSON Lines](https://jsonlines.org/) format, a widely used format for working with textual data. Users can also adapt and customize these modules to support data from arbitrary sources.

### Text cleaning and language identification[](#text_cleaning_and_language_identification)

After data acquisition but before further processing the data, an important step is to unify all the text into the Unicode format and identify the languages that are present throughout the acquired data. 

NeMo Curator uses the widely used [ftfy: fixes text for you](https://github.com/rspeer/python-ftfy) library to resolve all Unicode-related issues. NeMo Curator also provides helpers to identify the languages contained in every acquired document and organize them accordingly, which facilitates discarding irrelevant documents for LLM training.

### Quality filtering[](#quality_filtering)

NeMo Curator comes with a set of predefined qualitative criteria that are heuristics-based, as well as ML-based. Use the criteria to categorize documents into high– and low-quality buckets, enabling rapid dataset iteration and ensuring an expected level of quality from the acquired data. Customize these predefined criteria with configuration files to tune them to the individual business needs.

### Privacy filtering[](#privacy_filtering)

Compliance with data protection regulations is an important consideration for any enterprise solution. 

NeMo Curator provides a GPU-accelerated PII detection and redaction module. You can specify the categories to redact and how to redact them. For example, you could detect all names and addresses and replace them with other tokens.

### Domain and toxicity classification[](#domain_and_toxicity_classification)

Another aspect of ensuring data quality and relevance is to identify and remove out-of-domain, as well as toxic data. 

You can define custom filters to clean up your datasets and integrate them with external tools and [machine learning](https://www.nvidia.com/en-us/glossary/machine-learning/) models to classify the data into relevant and irrelevant categories.

### Deduplication[](#deduplication)

Internet-scale data can contain many identical or near-identical documents, which could incur storage and compute costs, and potentially degrade the model’s performance. 

NeMo Curator provides a configurable de-duplication module, which leverages highly optimized CUDA implementations of the MinHash and other commonly used algorithms to de-duplicate the documents.

### Streamlined scalability[](#streamlined_scalability)

NeMo Curator uses [Dask](https://www.dask.org/), an open-source and commercially friendly parallel computing library to easily scale across many CPUs and GPUs and accelerate every component of the data curation pipeline. 

NeMo Curator easily integrates with Dask data structures and supports Dask arrays, as well as [RAPIDS cuDF](https://github.com/rapidsai/cudf), to offload the processing to the correct resource with minimal intervention from developers. 

### Support for model customization tasks[](#support_for_model_customization_tasks)

In the near future, NeMo Curator will also support data curation for model customization tasks such as supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) approaches such as LoRA and P-tuning. 

NeMo Curator enables sampling and [blending various datasets](https://huggingface.co/datasets/nvidia/sft_datablend_v1) for SFT in [NeMo Aligner,](https://github.com/NVIDIA/NeMo-Aligner) which enables model customization and alignment with commercially permissible datasets to achieve near state-of-art model quality. 

## Enterprises harness NVIDIA AI for data curation [](#enterprises_harness_nvidia_ai_for_data_curation )

Leading AI companies and global enterprises are using NeMo Curator to accelerate data processing and to ensure that their training datasets are of high quality.

Hugging Face, the leading open platform for AI builders, is collaborating with NVIDIA to integrate NeMo Curator and accelerate DataTrove, their data processing pipeline for LLM training. “We are excited about the GPU acceleration capabilities of NeMo Curator and can’t wait to see them contributed to DataTrove!” says Jeff Boudier, product director at Hugging Face. 

“From dataset processing to AutoTrain powered by DGX Cloud, our new no-code service to easily fine-tune LLMs with the latest NVIDIA GPUs, our work with NVIDIA accelerates researchers and developers building their own AI.”

KT Corporation, the leading telecommunications company in South Korea, has started using NeMo Curator for scalability and high-quality dataset generation. KT is expecting state-of-the-art performance for LLMs trained on tokens prepared from NVIDIA NeMo Curator, which can generate high-quality datasets.

## Get started with NeMo Curator today [](#get_started_with_nemo_curator_today )

NeMo Curator is currently available under the Apache v2 license in the [/NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) GitHub repo.

Many of the features listed in this post will become available as a NeMo Curator microservice, which provides the easiest path for enterprises to get started with data curation from anywhere and offers streamlined performance and scalability to shorten the time to market. To apply, see [NeMo Curator Microservice Early Access](https://developer.nvidia.com/nemo-microservices-early-access).

As part of the early access program, you can also request access to other microservices, including NeMo Customizer and Evaluator, which can help simplify fine-tuning and assessment on custom generative AI models.