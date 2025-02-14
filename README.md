# 👀 About Animal-CLIP
Code release for "Animal-CLIP: A Dual-Prompt Enhanced Vision-Language Model for Animal Action Recognition"

Animal action recognition has a wide range of applications. With the rise of visual-language pretraining models (VLMs), new possibilities have emerged for action recognition. However, while current VLMs perform well on human-centric videos, they still struggle with animal videos. This is primarily due to the lack of domain-specific knowledge during model training and more pronounced intra-class variations compared to humans. To address these issues, we introduce Animal-CLIP, a specialized and efficient animal action recognition framework built upon existing VLMs. To address the lack of domain-specific knowledge in animal actions, we leverage the extensive expertise of large language models (LLMs) to automatically generate external prompts, thereby expanding the semantic scope of labels and enhancing the model's generalization capability. To effectively integrate external knowledge into the model, we propose a knowledge-enhanced internal prompt fine-tuning approach. We design a text feature refinement module to reduce potential recognition inconsistencies. Furthermore, to address the high intra-class variation in animal actions, this module generates adaptive prompts to optimize the alignment between text and video features, facilitating more precise partitioning of the action space. Experimental results demonstrate that our method outperforms six previous action recognition methods across three large-scale multi-species, multi-action datasets and exhibits strong generalization capability on unseen animals.

**Model structure:**
<img width="1161" alt="pipeline" src="https://github.com/user-attachments/assets/19712220-69d3-43b2-81bb-02721d0108ac" />

**Some prediction results:**

<img width="773" alt="image" src="https://github.com/user-attachments/assets/af443d3f-9110-4da1-b102-d12f9cc5eb65" />

## Data
You can access and download the [MammalNet](https://github.com/Vision-CAIR/MammalNet), [Animal Kingdom](https://github.com/sutdcv/Animal-Kingdom), [LoTE-Animal](https://github.com/LoTE-Animal/LoTE-Animal.github.io)  dataset to obtain the data used in the paper 
## Requirements
`pip install -r requirements.txt`
## Train
python -m torch.distributed.launch --nproc_per_node=<YOUR_NPROC_PER_NODE> main.py -cfg <YOUR_CONFIG> --output <YOUR_OUTPUT_PATH> --accumulation-steps 4 --description <YOUR_ACTION_DESCRIPTION_FILE> --animal_description <YOUR_ANIMAL_DESCRIPTION_FILE>
## Test
python -m torch.distributed.launch --nproc_per_node=<YOUR_NPROC_PER_NODE> main.py -cfg <YOUR_CONFIG> --output <YOUR_OUTPUT_PATH> --description <YOUR_ACTION_DESCRIPTION_FILE> --animal_description <YOUR_ANIMAL_DESCRIPTION_FILE> --only_test --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3 --resume <YOUR_MODEL_FILE>
## Acknowledgement
Thanks to the open source of the following projects:
[X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP),[BioCLIP](https://github.com/Imageomics/bioclip)
