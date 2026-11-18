# BioPAL experiment panel — 24 fits

Run **12 tasks × 2 checkpoints × 1 seed = 24 training fits total**. The panel uses **12 distinct checkpoints** across **7 registry sources**. Use seed `42` and one BioPAL adaptation configuration throughout. Each table row specifies two independent fits. Species-specific scores and uncertainty intervals reuse predictions from those fits.

| Modality | Task registry key | Checkpoint A | Checkpoint B | Prediction target | Primary metric |
| --- | --- | --- | --- | --- | --- |
| DNA | `plant_gen_splicing_arabidopsis_thaliana_acceptor` | `plant_caduceus_l20` | `caduceus_ps_131k` | Arabidopsis splice acceptor | Average precision |
| DNA | `nt_tasks_official_splice_sites_acceptors` | `plant_caduceus_l20` | `caduceus_ps_131k` | Human splice acceptor | Average precision |
| DNA | `nt_tasks_official_h3k27ac` | `ntv2_50m` | `dnabert2_117m` | Human histone-mark classification | Average precision |
| DNA | `fish_gue_ctcf` | `gena_lm_bert_base` | `dnabert2_117m` | Zebrafish CTCF binding | Average precision |
| DNA | `gen_lrb_bulk_rna_expression` | `caduceus_ps_131k` | `ntv3_100m_pre` | Tissue expression from genomic DNA | Mean squared error |
| RNA | `rna_dst_splice_site_acceptor` | `splicebert` | `rnafm` | Splice acceptor in pre-mRNA | Average precision, macro-averaged over test species |
| RNA | `rna_dst_ncrna_family_bnoise0` | `splicebert` | `rnafm` | Native multiclass ncRNA family | Macro-F1 |
| RNA | `rna_dst_mean_ribosome_load` | `splicebert` | `rnafm` | Ribosome load from a 5′ UTR | R² |
| Protein | `proteingym_kcne1_expression` | `esm2_t33_650m` | `ankh_base` | KCNE1 cell-surface expression | Spearman ρ |
| Protein | `proteingym_kcne1_function` | `esm2_t33_650m` | `ankh_base` | KCNE1 channel function | Spearman ρ |
| Protein | `cds_melting_point` | `esm2_t30_150m` | `prot_t5_xl_half` | Protein melting temperature | RMSE in target units |
| Protein | `proteingym_gfp_fluorescence` | `esm2_t30_150m` | `esm2_t33_650m` | GFP variant fluorescence | Spearman ρ |

## Comparison rules

- **DNA.** Run the two Caduceus checkpoints on both acceptor datasets to compare the plant- and human-pretrained models on the same biological endpoint. Check strand, junction position and central-motif handling before matching the human windows to the 398-bp plant span. Use the same crop for both models and label classes. On H3K27ac, NTv2 and DNABERT-2 compare two multispecies DNA models with different tokenizers. On fish CTCF, GENA-LM adds a human-pretrained comparator. Interpret rankings within each benchmark because organism, assay and negative sampling differ. [Plant](https://huggingface.co/datasets/InstaDeepAI/plant-genomic-benchmark) · [Human NT](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised) · [FishGUE](https://huggingface.co/datasets/bioinfoihb/Fish_GUE)
- **Bulk expression.** Use a single TSS-centered window length for both models. Target 8,192 bp. If curation retained shorter windows, fix one shared length from the available biological sequence before training. Use identical genes and target columns. Upstream supplies 218 outputs. If curation retained a scalar, identify the tissue or aggregation in the results. [Long-range benchmark](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark)
- **RNA.** SpliceBERT’s vertebrate pre-mRNA corpus and RNA-FM’s ncRNA corpus motivate reusing both models across splicing, family recognition and a quantitative 5′-UTR assay. For splicing, score the selected checkpoint separately on zebrafish, fly, worm and Arabidopsis. Report counts and prevalence per species. Preserve the distinct validation and test cohorts for ribosome load. [RNA benchmark and split definitions](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks)
- **Protein.** For KCNE1, join expression and function on substitution identity and full mutant sequence. Train separate endpoint models on the shared variants using identical partitions grouped by residue position and respecting official test assignments. Report the joined count. For melting point, verify the amino-acid translation of the upstream coding DNA and retain the protein-cluster split. Use each encoder’s residue tokenizer. GFP compares ESM-2 sizes on the same full mutant sequences and continuous fluorescence scores. [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) · [Melting-point benchmark](https://huggingface.co/datasets/InstaDeepAI/true-cds-protein-tasks) · [ProtT5 encoder](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)
- **Evaluation.** Resolve pinned revisions through `models.yaml` and curated inputs through `tasks.yaml`. Within each task, match examples, splits, biological span and adaptation budget. Select the checkpoint using validation data, then score the held-out test set. Report both scores and their paired difference. A paired bootstrap over held-out biological units can quantify test-sample uncertainty conditional on the fitted models. Training-seed variability remains unmeasured.
