# LoRA
> paper: https://arxiv.org/abs/2106.09685 \
> code: https://github.com/microsoft/LoRA\


## Case Study: mmseg-segformer-lora
1.  Datasets
    -   使用`mmseg_datasets/data2json.py`将img和seg文件路径转换成meta.json
    -   根据`train.jsonl`和`val.jsonl`定义数据集配置文件`mmseg_datasets/dmcode.py`
    -   根据hf的`AutoImageProcessor`和`nvidia/mit-b*`加载image_preocessor, 添加到train_transforms和val_transforms中

2. Pre-trained Model
    -   定义pre-trainde model的初始化方式, hf的segformer可以通过`SegformerFeatureExtractor`和`SegformerForImageClassification`使用`nvidia/mit-b*`进行初始化(可以正常训练); mmseg框架下训练的pth文件通过mmseg.api中的inference和init_model接口进行初始化(存在参数错误,不可以正常训练)
    -   根据`lora_config`文件, 使用`peft.get_peft_model`进行封装, 返回`lora_model`
        ```python
        self.lora_model = get_peft_model(self.model, self.lora_config)
        ```

3. Lora model training
    -   定义training_args和trainer
    -   ```python
        trainer.train()
        ```
    -   保存训练文件
        ```python
        self.lora_model.save_pretrained(output_dir+'/segformer-dmcode-lora')
        ```

4. Lora model inference
    -   加载image, 使用image_processor进行预处理
    -   加载Lora model
        ```
        inference_model = PeftModel.from_pretrained(self.model, model_id)
        ```
    -   调用lora model得到outputs和logits
    -   可视化结果
    
        