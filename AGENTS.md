# AGENTS.md

## Project Shape

- This is a `src/`-layout Python package (`deepsc_image`) for DeepSC-style image transmission: model/channel/loss/metrics code lives in `src/deepsc_image/`, configs live in `configs/`, and tests live in `tests/`.
- Main executable entrypoints are module CLIs, not top-level scripts: `python -m deepsc_image.train`, `python -m deepsc_image.evaluate`, `python -m deepsc_image.infer`, `python -m deepsc_image.smoke_test`, and `python -m deepsc_image.benchmark_training`.
- The GUI is Streamlit-only: run it with `streamlit run src/deepsc_image/app.py`; it reads `configs/gui.yaml` from the repo root.
- This repository is also the undergraduate thesis project workspace for `基于深度语义通信（DeepSC）的图像鲁棒传输系统设计与实现`; thesis source materials and generated thesis content belong under `paper/`.

## Thesis Writing Scope

- Treat this as a Chinese undergraduate thesis project, not a generic ML paper. Thesis prose must stay aligned with the implemented DeepSC-style image robust transmission system in this repository.
- Default thesis language is formal academic Chinese. Use third-person, objective technical wording; avoid casual phrasing and unsupported promotional claims.
- The core narrative is: a CNN-attention semantic encoder/decoder transmits low-dimensional image semantic features through differentiable wireless channels, then compares reconstruction quality and robustness against a lightweight JPEG baseline with GUI demonstration support.
- Do not silently broaden the thesis into unimplemented topics such as full BPG, LDPC/QAM chains, OFDM/MIMO hardware, Transformer/Swin backbones, diffusion decoding, cryptographic security, or measured energy saving unless new implementation/evidence is added.

## Paper Workspace

- All generated thesis drafts, outlines, chapter text, review notes, figure/table plans, citation working files, and writing checklists must be organized under `paper/`.
- Prefer Markdown drafts for agent collaboration unless the user explicitly asks for `.docx` output. If creating structured drafts, use clear names such as `paper/drafts/chapter_1_introduction.md` or `paper/thesis_draft.md`.
- Existing thesis materials include `paper/任务书.docx`, `paper/MinerU_markdown_任务书.md`, `paper/开题报告.docx`, `paper/MinerU_markdown_开题报告.md`, `paper/论文写作参考模板.docx`, `paper/MinerU_markdown_论文写作参考模板.md`, `paper/deep-research-report.md`, `paper/参考论文/`, and `paper/参考文献/`.
- Before writing or revising thesis content, read the relevant Markdown extraction first. Use the `.docx` files only when layout fidelity or original formatting must be checked.
- All files containing Chinese text must be read and written as UTF-8. When using PowerShell or scripts, explicitly choose UTF-8 encoding and avoid legacy ANSI/GBK defaults that can corrupt Chinese content.

## Thesis Skill Usage

- Skills under `.opencode/skills/` are treated as thesis/research-writing support skills for this project. When a thesis, literature review, experiment analysis, figure/table, presentation, or academic writing task overlaps with any of them, load the relevant skill before drafting or delegating.
- For thesis prose, documentation collaboration, `.docx` work, PDF reading/review, static visual design, figure/poster creation, and human-style polishing, proactively use relevant skills such as `doc`, `doc-coauthoring`, `pdf`, `canvas-design`, `academic-plotting`, and `humanizer`.
- Prefer project thesis skills over generic writing behavior. If a task involves thesis content and a matching skill exists, using the skill is part of the expected workflow, not optional.

## Thesis Source Priority

- Evidence priority for thesis writing is: task book/opening report/template materials in `paper/`; repository code/configs/tests/README; real experiment artifacts under `outputs/`; reference papers under `paper/参考论文/` and `paper/参考文献/`; then newly searched external sources.
- Never invent citations, experiment results, datasets, checkpoints, ablation studies, latency numbers, PSNR/SSIM values, or implementation details.
- Claims about model design must map to code such as `src/deepsc_image/model.py`, `channels.py`, `losses.py`, `metrics.py`, `baseline.py`, `app.py`, and `configs/*.yaml`.
- Claims about training or evaluation must map to actual configs, commands, checkpoints, logs, `history.*`, `summary.json`, `metrics.json`, generated plots, or benchmark JSON files.
- If evidence is missing, write an explicit placeholder such as `[待补充实验结果：AWGN 0 dB PSNR/SSIM 对比]` rather than fabricating content.

## Thesis Requirements From Materials

- The task book and opening report require an image semantic communication system based on DeepSC/JSCC ideas, with a convolution-attention semantic encoder and decoder.
- The system must include differentiable AWGN and Rayleigh channel models and evaluate robustness over SNR settings that cover at least `-5 dB` to `20 dB` where applicable.
- The software demonstration must support local image upload, preprocessing, SNR adjustment, original image / DeepSC reconstruction / traditional baseline comparison, and automatic PSNR/SSIM display.
- Training and evaluation discussion should mention CIFAR-10 for training or quick verification, Kodak for image-quality evaluation, Adam optimization, and `MSE + (1 - SSIM)` mixed loss when supported by code/configs.
- Experimental analysis should cover reconstruction quality, robustness across SNR/channel types, compression or semantic channel effects, latency when measured, loss convergence, and visual comparison.
- Non-technical analysis should cautiously discuss privacy/security and green communication. Do not claim cryptographic security or proven energy savings unless measured evidence exists.

## Chinese Thesis Structure

- Follow `paper/MinerU_markdown_论文写作参考模板.md` for thesis structure and formatting expectations.
- Chinese abstract should be about 400-600 Chinese characters and include purpose/significance, problem, method, results, and conclusion. English abstract must be consistent with the Chinese abstract.
- Provide 3-5 keywords, arranged from broad to narrow when possible, separated by semicolons.
- A suitable chapter flow for this project is: introduction; related work and theory; system requirements and overall design; model/channel/loss/metric implementation; experiments and result analysis; GUI and system testing; privacy/security and green communication discussion; conclusion and outlook.
- Each chapter should end with a brief chapter summary when appropriate. Do not duplicate the abstract in the introduction.


## Thesis Generation Workflows

- When asked to generate a thesis outline, produce a project-specific undergraduate thesis outline rather than a generic academic-paper skeleton. The outline must follow `paper/MinerU_markdown_论文写作参考模板.md`, keep `# 中文摘要`, `# 英文摘要`, the introduction/chapters, conclusion/outlook, and `# 参考文献` in a logical final order, and map this repository's DeepSC-style image transmission modules, configs, datasets, GUI, and experiment artifacts to the appropriate chapters.
- Outline drafts may use Markdown headings for structure, but should not use bullet lists unless the user asks for a checklist. Chapter and subsection numbering should be consistent with the intended thesis structure, such as `## 1.1 研究背景与意义` and `### 1.1.1 语义通信研究背景`, and no subsection numbering should appear before the first numbered chapter.
- When expanding an outline into a chapter or section, first identify the current section's role in the full thesis, available upstream/downstream context, evidence sources, expected figures/tables/equations, and citation needs. Write only the requested section, keep it aligned with existing chapter context, and do not restate the whole outline unless the user asks.
- Chapter drafts should be written as thesis prose directly in Markdown files under `paper/`, without wrapping the content in code fences and without using template placeholders such as `{topic}` or `{outline}` in final text. If a target word count is provided, treat it as an approximate scope guide while preserving accuracy and readability.
- Abstracts should normally be drafted after the core outline and key experimental evidence are known. The Chinese abstract should cover research purpose/significance, problem, method, results, and conclusion without inventing metrics; the English abstract must faithfully match the Chinese abstract. Keywords should be 3-5 items for this thesis unless the user explicitly requests another count, separated by semicolons.
- If drafting a standalone section that uses citations before the final bibliography is complete, append a temporary `参考文献` block at the end of that section and number only the references actually cited in that draft, in first-appearance order. Temporary entries must still correspond to real verified sources from `paper/参考文献/`, `paper/参考论文/`, `paper/deep-research-report.md`, or a verified external source; otherwise use an explicit `[需要补充文献：...]` placeholder instead of fake bibliographic data.
- For final integrated thesis drafts, merge temporary section references into one GB/T 7714-style bibliography, renumber citations globally by first appearance, and remove duplicate or unused references.

## Thesis Prose Style

- Thesis body text should be output as coherent paragraphs by default, not as bullet lists, unless the user explicitly asks for outlines, tables, or checklists.
- Adjacent body paragraphs should differ noticeably in length, with a difference of more than 100 Chinese characters where feasible. This requirement is important for reducing mechanical rhythm in long-form thesis prose.
- Use a light academic style: accurate, restrained, and easy to read. Avoid overly dense theory stacking, exaggerated claims, and sentences that are too long.
- Paragraph openings should vary naturally according to context. Do not make adjacent paragraphs into parallel sentence patterns or repeated rhetorical structures.
- Do not use Chinese or English quotation marks in thesis body prose unless quoting a verified source or preserving an official title that requires them.
- Use Chinese punctuation throughout Chinese thesis prose, including commas, parentheses, colons, semicolons, and question marks. Do not insert spaces between Chinese and English terms, numbers, or abbreviations unless a format standard explicitly requires it.
- Avoid formulaic transition words such as 首先、其次、最后、然而、综上所述、此外、具体而言. Choose context-specific transitions instead, or omit the transition when the paragraph relation is already clear.
- Preserve technical correctness and thesis-context alignment. Do not simplify into statements that create conceptual errors, unsupported causal claims, or obvious AI-generated writing traces.

## Formatting And Numbering

- Use chapter-based numbering for figures, tables, and equations, such as `图 3.1`, `表 4.2`, and `(2.1)`, unless the final school template requires a different separator.
- Every figure and table must be mentioned in the text before or near its appearance, for example `如图 4.1 所示` or `见表 4.2`.
- Figures need self-contained captions. Experimental plots should label axes with quantity, symbol, and unit where applicable, such as `SNR/dB`, `PSNR/dB`, `SSIM`, or `Latency/s`.
- Tables should be concise, preferably three-line style in final formatting, with consistent decimal precision and clear metric directions where useful.
- References should follow GB/T 7714 sequential citation style. Citation numbers in text must follow first appearance order.
- Maintain low-error formatting: check typos, figure/table numbering, reference ordering, unit notation, equation symbols, and consistency of abbreviations.

## Citation And Integrity Rules

- Do not hallucinate references. Only cite a paper after reading a reliable source from `paper/参考文献/`, `paper/参考论文/`, `paper/deep-research-report.md`, or a verified external source.
- Do not fabricate author names, titles, venues, years, DOIs, page ranges, arXiv IDs, or GB/T 7714 entries.
- If a citation is needed but not verified, mark it explicitly, for example `[需要补充文献：DeepJSCC wireless image transmission]`.
- Paraphrase with attribution. Do not copy long passages from the task book, opening report, template, research report, or papers into the thesis without quotation/citation handling.
- The template notes plagiarism checks and academic integrity requirements. Keep copied wording minimal and transform source material into original explanation grounded in this project.
- `paper/deep-research-report.md` is useful for research direction and candidate references, but its web-style citation markers are not final thesis citations. Convert only verified sources into GB/T 7714 references.

## Thesis-Code Traceability

- Map `SemanticEncoder`, `SemanticDecoder`, SE attention, spatial attention, and `DeepSCImageModel` to the semantic codec design chapter.
- Map AWGN/Rayleigh implementations in `channels.py` to the differentiable wireless channel modeling chapter.
- Map the mixed reconstruction loss in `losses.py` to the training objective section; describe the actual formula used by the code.
- Map PSNR/SSIM functions in `metrics.py` to the evaluation metric section.
- Map `baseline.py` to the lightweight JPEG baseline. The current project intentionally avoids external BPG/system codec dependencies for Windows/CPU usability.
- Map `train.py`, `evaluate.py`, `infer.py`, `benchmark_training.py`, and `configs/*.yaml` to reproducible experiment setup.
- Map `app.py` and `configs/gui.yaml` to the Streamlit GUI system design and testing sections.

## Thesis Task Workflow

- Before drafting a section, identify the target chapter, thesis requirement it supports, source materials to read, code/output evidence needed, expected figures/tables/equations, and required citations.
- For literature review, start from `paper/deep-research-report.md` and verified PDFs in `paper/参考文献/`; organize related work by theme rather than listing papers one by one.
- For experiment sections, first inspect or generate real artifacts. Use project commands to produce missing metrics/plots when feasible; otherwise leave placeholders for the user to run experiments.
- After writing, self-check for unsupported claims, citation placeholders, chapter structure, figure/table/equation numbering, terminology consistency, and alignment with task-book requirements.
- Keep thesis drafts separate from code changes. Do not modify datasets, checkpoints, or experiment outputs just to make prose easier to write.

## Setup And Imports

- Install editable mode before running module commands or tests: `pip install -r requirements.txt` then `pip install -e .`. Tests need pytest, so use `pip install -e ".[dev]"` if pytest is missing.
- If editable install is skipped, module imports depend on setting `PYTHONPATH=src`; prefer editable install instead of baking path hacks into code.
- `device: auto` resolves to CUDA when available and CPU otherwise; CPU must keep working for smoke tests and GUI/inference pipeline checks.

## Verification Commands

- Fast no-dataset smoke check: `python -m deepsc_image.smoke_test`; success prints `SMOKE_TEST_OK ...`.
- Unit tests: `pytest`. For focused checks use paths such as `pytest tests/test_cli_config.py` or `pytest tests/test_training_speed_config.py`.
- Quick training connectivity uses the real CIFAR config but can be shortened: `python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml --epochs 1`.
- Evaluation writes artifacts with: `python -m deepsc_image.evaluate --config configs/eval_kodak.yaml --checkpoint outputs/train_cifar10_awgn/<experiment_dir>/best_model.pth`.
- Benchmarking requires a single semantic channel when the YAML contains a list: add `--semantic-channels 16` to `python -m deepsc_image.benchmark_training ...`.
- For thesis claims, verify against source materials, code/configs, and real output artifacts before writing numerical or comparative conclusions.

## Config And Runtime Gotchas

- Training/evaluation configs are YAML-first; CLI flags override nested YAML values and `--interactive` changes only the in-memory run config, not files in `configs/`.
- Default CIFAR configs use `dataset.download: false`; training expects CIFAR-10 under `datasets/cifar10/cifar-10-batches-py` unless `--download` or YAML is changed.
- `configs/eval_kodak.yaml` expects image files in `datasets/kodak` and writes `metrics.json`, `psnr_vs_snr.png`, and `ssim_vs_snr.png` under `outputs/eval_kodak` by default.
- A list in `model.semantic_channels` triggers one training output subdirectory per value while sharing a timestamp; generated run dirs include channel, SNR, semantic channels, image size, and seed.
- Training writes `best_model.pth`, `last_model.pth`, `config.yaml`, `history.csv`, `history.json`, `loss_curve.png`, and `summary.json`; there are no per-epoch `epoch_*.pth` checkpoints.
- `best_model.pth` is selected by lowest training loss because there is no validation split; do not describe it as validation-best.
- Missing checkpoints are allowed for GUI/infer/evaluate pipeline validation but mean random-initialized model quality; never use those results as DeepSC performance evidence.
- Checkpoint loading is intentionally restricted to raw state dicts or this project's payload keys (`model_state`, `epoch`, `config`, `bandwidth_estimate`); avoid adding unsafe `torch.load` paths.
- AMP is only effective on CUDA even if requested; `persistent_workers` and `prefetch_factor` are only active when `training.dataloader.num_workers > 0`.

## Data And Artifacts

- Thesis drafts and writing notes belong in `paper/`; experiment metrics, plots, checkpoints, and runtime artifacts belong in `outputs/` and must preserve existing experiment provenance.
- Do not assume `datasets/` and `outputs/` are untracked just because README says large artifacts are excluded; this repo currently tracks datasets and several output/checkpoint artifacts.
- Before deleting, rewriting, or committing anything under `datasets/` or `outputs/`, check `git ls-files`/`git status` and preserve user-generated experiment results unless explicitly told otherwise.
- `.gitignore` currently ignores caches, virtualenvs, `.streamlit/secrets.toml`, and `.sisyphus/`, but not `datasets/` or `outputs/`.

## Code Conventions

- Keep the JPEG baseline dependency-light in `baseline.py`; the current project intentionally avoids external BPG/system codec requirements for Windows/CPU usability.
- Keep plotting/artifact generation dependency-light: training/evaluation curves use matplotlib with a headless backend.
- When changing model dimensions or checkpoint handling, update GUI loading tests because `app.load_model` reads embedded checkpoint config to construct the matching model before loading weights.
