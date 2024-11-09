import torch
import evaluate
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel
from torchvision.transforms import ColorJitter
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot

from transformers import AutoImageProcessor
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import SegformerFeatureExtractor, SegformerForImageClassification


class MMSegSegformerLoRA:
    def __init__(self, segformer_checkpoints, segformer_config, lora_config, image_processor_config, device='cpu'):
        self.segformer_checkpoints = segformer_checkpoints
        self.segformer_config = segformer_config
        self.lora_config = lora_config
        self.image_processor_config = image_processor_config

        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(self.image_processor_config, do_reduce_labels=True)
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        self.metric = evaluate.load("mean_iou")

        self.id2label = {0:"background", 1:"target"}

    def load_model(self):
        # import json
        # from huggingface_hub import cached_download, hf_hub_url
        # repo_id = "huggingface/label-files"
        # filename = "ade20k-id2label.json"
        # id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
        # id2label = {int(k): v for k, v in id2label.items()}
        # label2id = {v: k for k, v in id2label.items()}
        # checkpoint = "nvidia/mit-b0"
        # self.model = AutoModelForSemanticSegmentation.from_pretrained(
        #     checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        # )
        self.model = init_model(self.segformer_config, self.segformer_checkpoints, device=self.device)
        if self.device == 'cpu':
            self.model = revert_sync_batchnorm(self.model)
        self.print_trainable_parameters(self.model)
        self.lora_model = get_peft_model(self.model, self.lora_config)
        self.print_trainable_parameters(self.lora_model)

        for name, param in self.lora_model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def load_data(self, data_script, ):
        datasets = load_dataset(data_script)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['validation']

        self.train_dataset.set_transform(self.train_transforms)
        self.val_dataset.set_transform(self.val_transforms)


    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")
      
    def handle_grayscale_image(self, image):
        np_image = np.array(image)
        if np_image.ndim == 2:
            tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
            return Image.fromarray(tiled_image)
        else:
            return Image.fromarray(np_image) 
    
    def train_transforms(self, example_batch):
        images = [self.jitter(self.handle_grayscale_image(x)) for x in example_batch["image"]]
        labels = [x.convert('L') for x in example_batch["label"]]
        inputs = self.image_processor(images, labels)
        return inputs


    def val_transforms(self, example_batch):
        images = [self.handle_grayscale_image(x) for x in example_batch["image"]]
        labels = [x for x in example_batch["label"]]
        inputs = self.image_processor(images, labels)
        return inputs

    def compute_metrics(self, eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            # currently using _compute instead of compute
            # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
            metrics = self.metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(self.id2label),
                ignore_index=0,
                reduce_labels=self.image_processor.do_reduce_labels,
            )

            per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            metrics.update({f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
            metrics.update({f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)})

            return metrics

    def process(self, output_dir):
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=5e-4,
            num_train_epochs=50,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            save_total_limit=3,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=5,
            remove_unused_columns=False,
            push_to_hub=False,
            label_names=["labels"],
        )

        trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        self.lora_model.save_pretrained(output_dir+'/segformer-dmcode-lora')
    
    def inference(self, image_path, model_id):
        image = Image.open(image_path)
        encoding = self.image_processor(image.convert('RGB'), return_tensors='pt')
        inference_model = PeftModel.from_pretrained(self.model, model_id)
        with torch.no_grad():
            outputs = inference_model(pixel_values=encoding.pixel_values)
            logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
            )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        self.visualize_pre(pred_seg=pred_seg)
    
    def visualize_pre(self, pred_seg):
        color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
        palette = np.array([[0,0,0],[255,255,255]])
        for label, color in enumerate(palette):
            color_seg[pred_seg == label, :] = color
            color_seg = color_seg[..., ::-1]  # convert to BGR

        plt.figure(figsize=(15, 10))
        plt.imshow(color_seg)
        plt.show()

        


if __name__ == '__main__':
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "in_proj", "out_proj"
            # "query", "value"
            ],
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["decode_head"],
    )

    mmseg_segformer_lora_obj = MMSegSegformerLoRA(segformer_checkpoints='/Users/yxizhong/workspace/DMCode/output/mmseg/segformer/files/segformer_pretraind_iter_16000.pth',
                                                  segformer_config='/Users/yxizhong/workspace/DMCode/output/mmseg/segformer/pre-trained/segformer_mit-b2_8xb2-160k_ade20k-512x512.py',
                                                  image_processor_config='/Users/yxizhong/workspace/DMCode/output/mmseg/segformer/files/config.json',
                                                  lora_config=lora_config)
    
    mmseg_segformer_lora_obj.load_data(data_script='/Users/yxizhong/workspace/visual-tuning/lora/mmseg_datasets/dmcode.py')
    mmseg_segformer_lora_obj.load_model()
    mmseg_segformer_lora_obj.process(output_dir='/Users/yxizhong/workspace/visual-tuning/lora/output')
