# VITS-Pytorch
基于Pytorch实现的VITS

1. 准备数据

```shell
mkdir dataset
cd dataset
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/sampled_audio4ft_v2.zip
unzip sampled_audio4ft_v2.zip
```

2. 下载预训练模型文件

```shell
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_model/g_net.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_model/d_net.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/configs/uma_trilingual.json -O ./pretrained_model/configs.json
```

然后执行下面代码转换格式
```python
import json

import yaml

with open('pretrained_model/config.json', "r") as f:
    data = f.read()
config = json.loads(data)

with open('configs/config.yml', 'w', encoding='utf-8') as f:
    yaml_datas = yaml.dump(config, indent=2, sort_keys=False, allow_unicode=True)
    f.write(yaml_datas)
```

3. 制作数据列表

```shell
python preprocess_data.py
```

4. 开始训练

```shell
python train.py
```