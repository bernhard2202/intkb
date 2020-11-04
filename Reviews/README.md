# COLING 2020 Reviews for Submission #1539

Title: IntKB: A Verifiable Interactive Framework for Knowledge Base Completion
Authors: Bernhard Kratzwald, Guo Kunpeng, Stefan Feuerriegel and Dennis Diefenbach

---------------------------------------------------------------------------
## REVIEWER #1

### Reviewer's Scores
                         Relevance (1-5): 5
               Readability/clarity (1-5): 4
                       Originality (1-5): 4
   Technical correctness/soundness (1-5): 3
                   Reproducibility (1-5): 3
                         Substance (1-5): 3

### Detailed Comments
The paper proposed an interactive framework for KB completion task from text based on question answering pipeline. Given a query with head entity and relation type, they first annotate the tail entity based on the QA pipeline with three modules, including sentence selection, relation extraction and answer re-ranking. Then the answer-triggering module is used to decide if the human interaction will be invoked and the human annotations will be used to update the KB. The proposed framework is evaluated with extensive results.

Strengths:
1. The paper solves the KB completion task based on the question answering pipeline, which leverages the answers obtained from the QA task to include new facts in KB.
2. The human interactions can be involved in the framework, which enables humans to verify the new identified facts.
3. Extensive results are provided

Weaknesses:
1. The proposed method is only compared with BERT-sentence and Naive QA pipeline, however, there are many other works about KB completion using other techniques, including link prediction or triplet classification. It is better to compare with these methods.
2. I believe that the new facts identified by different methods tend to be different. I suggest the authors provide some examples and provide the analysis about this difference.
3. The paper mentions that the human feedback is requested sparsely, however, the evaluation in section 7.3 involves more than 18k user interactions. Is this done by simulation or by human interactions in practice? This is not clear to me. If it is performed by simulation, I think it cannot be claimed as human interactions. If it is done by human annotations, it requires lots of human efforts. Also, how many annotators are involved in this interaction step? How many annotations do you have for each new fact, and how do you align the disagreement annotations for it? The authors need to discuss more details about this since the human interaction is claimed as one of the main contributions in the paper.



### Reviewer's Scores
            Overall recommendation (1-5): 3
                        Confidence (1-5): 4
                       Presentation Type: Poster
     Recommendation for Best Paper Award: No


---------------------------------------------------------------------------
## REVIEWER #2

### Reviewer's Scores
                         Relevance (1-5): 5
               Readability/clarity (1-5): 5
                       Originality (1-5): 3
   Technical correctness/soundness (1-5): 3
                   Reproducibility (1-5): 3
                         Substance (1-5): 2


### Detailed Comments

This article proposes IntKB, a novel interactive framework for KB completion from text based on a question answering pipeline, which is tailored to the specific needs of a human-in-the-loop paradigm. In practice, IntKB shows facts along textual evidence in order to facilitate human verification and source tracing.

The research point of this article is highly relevant to COLING, and it’s well-written and well-structured, which makes it easy to understand. The logic of this article is coherent.

However, the experiment part may not be convincing enough. The comparison with two baselines are not conducted on the same training datasets, which makes the experiment lack fairness. I suggest that more experiments could be done to compare the performance of IntKB with existing KB completion algorithms from free text.

There are some issues with the proposed human interaction. Experiments show there are some performance drops with more human feedback (see Figs 2.). More analysis on these unusual drops should be added.


### Reviewer's Scores
            Overall recommendation (1-5): 3
                        Confidence (1-5): 3
                       Presentation Type: Poster
     Recommendation for Best Paper Award: No


---------------------------------------------------------------------------
## REVIEWER #3

### Reviewer's Scores
                         Relevance (1-5): 5
               Readability/clarity (1-5): 4
                       Originality (1-5): 3
   Technical correctness/soundness (1-5): 3
                   Reproducibility (1-5): 3
                         Substance (1-5): 3

### Detailed Comments
This paper proposes to systematically tackle the KB completion problem with a verifiable interactive framework based on a QA pipeline. Several challenges are reasonably addressed in the framework, namely, sentence selection, relation extraction, answer re-ranking, answer triggering, and entity linking. Experiments on a constructed dataset from wikidata and wikipedia demonstrate its effectiveness. The paper is generally well organized and easy to follow.

Strengths:
1. The authors present a systematic solution for interactive kb completion. Major sub-problems are also carefully addressed accordingly.
2. The idea of using question answering pipeline to obtain a small set of answer entities for human verification can substantially reduce the demanded human efforts, making the design realistic.
3. Fine-tuning BERT after fact alignment may improve the performance of relation extraction step, hence enable the continuous learning paradigm. Experiments show that this paradigm is of great potentials in terms of improving the prediction performance on unseen relations.

Weaknesses:
1. As a systematic solution for interactive KB completion tasks, there are seven modules in total co-operating to fulfill the task. It inevitably involves a lot efforts in coding and data processing to achieve a good performance, which may arouse concerns about the reproducibility of this work.
2. I did not quite get the settings of the interaction simulation. The authors claim they do not use the ground-truth labels but only approve correct labels and reject incorrect labels. But if this makes a difference, does that mean correct labels can also be rejected during the simulation? How exactly the simulation procedure is implemented?
3. More ablation analysis, such as the performance of each component, could be helpful to better understand the potentials and the limitations (and the bottleneck if there is) of the framework. Besides, component-wise performance helps verify whether the ultimate performance reported is convincible.



### Reviewer's Scores
            Overall recommendation (1-5): 4
                        Confidence (1-5): 3
                       Presentation Type: Poster
     Recommendation for Best Paper Award: No