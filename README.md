# MAACA: Multimodal Aspect-Aware Complaint Analysis for E-commerce Video-based Product Reviews

This repo is the official implementation and dataset for the ACM Multimedia 2024 paper ["Seeing Beyond Words: Multimodal Aspect-Level Complaint
Detection in Ecommerce Videos"](https://openreview.net/pdf/1ef5082841dc737c3f0dfbc4617fc60c8c2d59a3.pdf).

## Introduction
- We introduce the Video Complaint Dataset (VCD), a novel resource aimed at advancing research in aspect-level complaint
detection.
- We propose a Multimodal Aspect-Aware Complaint Analysis
(MAACA) framework for aspect-level complaint detection
from discourse (ACDD). MAACA extends the [ALPRO](https://github.com/salesforce/ALPRO) pretraining strategy, to incorporate the audio modality into
its architecture as well as in its pre-training strategy. Furthermore, MAACA incorporates a moment retrieval step, augmenting the identification of pertinent segments within the
video clip crucial for the accurate detection of fine-grained
aspect categories and aspect-level complaints.
- We propose a gated-fusion mechanism to efficiently integrate
multimodal representations while considering the varying
importance of each feature through a gating mechanism.
- Extensive experiments conducted on the VCD dataset demonstrate the significant superiority of our framework over existing multimodal baselines, providing valuable insights into
the application of multimodal representation learning frameworks for downstream tasks.

## Model Architecture

![ACM Multimedia Model](images/ACMM_Model.png)
  
## Dataset
The Video Complaint Dataset (VCD) is a novel resource aimed at advancing research in aspect-level complaint detection from video. 
The CSV files containing the train and test split used in the paper is made available under the `data` folder.
The audio and corresponding video for each video clip can be obtained by running the `download_data.sh` script.
The corresponding moment retrieved timestamps are obtained from running CGDETR model on the video clips.

## How to Run


## Citation

If you find this repo useful, please cite our paper:

```bibtex
@inproceedings{maaca,
  title={Seeing Beyond Words: Multimodal Aspect-Level Complaint Detection in Ecommerce Videos},
  author={Anonymous},
  booktitle={Anonymous},
  year={2024}
}
```