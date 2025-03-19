# Torch-DirectML

Official MS Documentation: <a href="https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows">Enable PyTorch with DirectML on Windows</a>

It looks like we may be able to test some things at near native speeds using the RX 9070 XT in PyTorch via DirectML. This should technically work but I have never used DirectML for PyTorch models/training.

```
# Setup the environment
conda create -n directml python=3.12 -y
conda activate directml

# Make Jupyter folder
mkdir Jupyter-DirectML
cd Jupyter-DirectML

# Install Torch-DirectML
pip install torch-directml jupyterlab

# Start a Jupyter Lab
jupyter lab password
jupyter lab --ip 10.0.0.35 --port 8888 --no-browser
```

Connect via VSCode or open the url in the console output. Start a notebook and run the following:

```
import torch
import torch_directml
dml = torch_directml.device()
print(dml)
```

This 100% locked up my workstation, testing just the `import torch` module:

```
import torch
```

Great success.

Adding the `import torch_directml` module:

```
import torch
import torch_directml
```

Great success.

Setting a var to the device `dml = torch_directml.device()`

```
import torch
import torch_directml
dml = torch_directml.device()
```

Great success.

Ok everthing seems to be working, this might have been a fluke.

Trying to print what device is contained in the var dml `print(dml)`:

```
import torch
import torch_directml
dml = torch_directml.device()
print(dml)
```

This works and the output is: `privateuseone:0`

Let's test this on one of my existing ipynb's. This will require some packages to be installed and some tweaks to the code.

Install matplotlib, tqdm: `pip install matplotlib tqdm`

This is working but its only offloading to the GPU a small amount. The GPU will load from 3% to 15-17% but the cpu will go to 65-75% so Im not sure what is happening. This will require some research as this is my literal first time trying out DirectML in PyTorch.

Thoughts: Drivers, Pytorch config, maybe we need some DirectML SDK features. It could be that the 9070 XT is to new and just isnt supported in this capacity, DirectML definitely works for Stable Diffusion via Amuse so this leads me to think there is an issue feeding the GPU with/or configs.

In dataloader these options reduced memory footprint and cpu load but also reduced gpu load.
`pin_memory=False, num_workers=16`

Looks like I'm loading data to ram and not vram.

This needed a few specific packages/modules installed but now appears to be working with the official MS DirectML Stable Diffusion Jupyter Notebook. After generating a few images it would appear I ran out of VRAM, so the ram was not being purged. This does work though.

```
(directml) C:\Users\phill\Jupyter-DirectML>pip list
Package                   Version
------------------------- ---------------
aiofiles                  23.2.1
altair                    5.5.0
annotated-types           0.7.0
anyio                     4.9.0
argon2-cffi               23.1.0
argon2-cffi-bindings      21.2.0
arrow                     1.3.0
asttokens                 3.0.0
async-lru                 2.0.5
attrs                     25.3.0
babel                     2.17.0
beautifulsoup4            4.13.3
bleach                    6.2.0
certifi                   2025.1.31
cffi                      1.17.1
charset-normalizer        3.4.1
click                     8.1.8
colorama                  0.4.6
comm                      0.2.2
contourpy                 1.3.1
cycler                    0.12.1
debugpy                   1.8.13
decorator                 5.2.1
defusedxml                0.7.1
diffusers                 0.32.2
executing                 2.2.0
fastapi                   0.115.11
fastjsonschema            2.21.1
ffmpy                     0.5.0
filelock                  3.18.0
fonttools                 4.56.0
fqdn                      1.5.1
fsspec                    2025.3.0
gradio                    3.48.0
gradio_client             0.6.1
groovy                    0.1.2
h11                       0.14.0
httpcore                  1.0.7
httpx                     0.28.1
huggingface-hub           0.29.3
idna                      3.10
importlib_metadata        8.6.1
importlib_resources       6.5.2
ipykernel                 6.29.5
ipython                   9.0.2
ipython_pygments_lexers   1.1.1
isoduration               20.11.0
jedi                      0.19.2
Jinja2                    3.1.6
json5                     0.10.0
jsonpointer               3.0.0
jsonschema                4.23.0
jsonschema-specifications 2024.10.1
jupyter_client            8.6.3
jupyter_core              5.7.2
jupyter-events            0.12.0
jupyter-lsp               2.2.5
jupyter_server            2.15.0
jupyter_server_terminals  0.5.3
jupyterlab                4.3.6
jupyterlab_pygments       0.3.0
jupyterlab_server         2.27.3
kiwisolver                1.4.8
markdown-it-py            3.0.0
MarkupSafe                2.1.5
matplotlib                3.10.1
matplotlib-inline         0.1.7
mdurl                     0.1.2
mistune                   3.1.3
mpmath                    1.3.0
narwhals                  1.31.0
nbclient                  0.10.2
nbconvert                 7.16.6
nbformat                  5.10.4
nest-asyncio              1.6.0
networkx                  3.4.2
notebook_shim             0.2.4
numpy                     1.26.4
orjson                    3.10.15
overrides                 7.7.0
packaging                 24.2
pandas                    2.2.3
pandocfilters             1.5.1
parso                     0.8.4
pillow                    10.4.0
pip                       25.0
platformdirs              4.3.6
prometheus_client         0.21.1
prompt_toolkit            3.0.50
psutil                    7.0.0
pure_eval                 0.2.3
pycparser                 2.22
pydantic                  2.10.6
pydantic_core             2.27.2
pydub                     0.25.1
Pygments                  2.19.1
pyparsing                 3.2.1
python-dateutil           2.9.0.post0
python-json-logger        3.3.0
python-multipart          0.0.20
pytz                      2025.1
pywin32                   310
pywinpty                  2.0.15
PyYAML                    6.0.2
pyzmq                     26.3.0
referencing               0.36.2
regex                     2024.11.6
requests                  2.32.3
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rich                      13.9.4
rpds-py                   0.23.1
ruff                      0.11.0
safehttpx                 0.1.6
safetensors               0.5.3
scipy                     1.15.2
semantic-version          2.10.0
Send2Trash                1.8.3
setuptools                75.8.0
shellingham               1.5.4
six                       1.17.0
sniffio                   1.3.1
soupsieve                 2.6
stack-data                0.6.3
starlette                 0.46.1
sympy                     1.13.3
terminado                 0.18.1
tinycss2                  1.4.0
tokenizers                0.21.1
tomlkit                   0.13.2
torch                     2.4.1
torch-directml            0.2.5.dev240914
torchvision               0.19.1
tornado                   6.4.2
tqdm                      4.67.1
traitlets                 5.14.3
transformers              4.49.0
typer                     0.15.2
types-python-dateutil     2.9.0.20241206
typing_extensions         4.12.2
tzdata                    2025.1
uri-template              1.3.0
urllib3                   2.3.0
uvicorn                   0.34.0
wcwidth                   0.2.13
webcolors                 24.11.1
webencodings              0.5.1
websocket-client          1.8.0
websockets                11.0.3
wheel                     0.45.1
zipp                      3.21.0
```

```
import torch
import torch_directml
import gradio as gr
from diffusers import AutoPipelineForText2Image,  StableDiffusionPipeline, LMSDiscreteScheduler
from PIL import Image
import numpy as np
 
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.
 
lms = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)
 
device = torch_directml.device(torch_directml.default_device())
 
block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")
num_samples = 2
 
def load_model(model_name):
    return AutoPipelineForText2Image.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
 
model_name = "stabilityai/sd-turbo"
pipe = load_model("stabilityai/sd-turbo")
 
def infer(prompt, inference_step, model_selector):
    global model_name, pipe
 
    if model_selector == "SD Turbo":
        if model_name != "stabilityai/sd-turbo":
            model_name = "stabilityai/sd-turbo"
            pipe = load_model("stabilityai/sd-turbo")
    else:
        if model_name != "stabilityai/sdxl-turbo":
            model_name = "stabilityai/sdxl-turbo"
            pipe = load_model("stabilityai/sdxl-turbo")
        
    images = pipe(prompt=[prompt] * num_samples, num_inference_steps=inference_step, guidance_scale=0.0)[0]
    return images
 
 
with block as demo:
    gr.Markdown("<h1><center>Stable Diffusion Turbo and XL Turbo with DirectML Backend</center></h1>")
 
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
 
                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                iteration_slider = gr.Slider(
                    label="Steps",
                    step = 1,
                    maximum = 4,
                    minimum = 1,
                    value = 1         
                )
 
                model_selector = gr.Dropdown(
                    ["SD Turbo", "SD Turbo XL"], label="Model", info="Select the SD model to use", value="SD Turbo"
                )
 
        gallery = gr.Gallery(label="Generated images", show_label=False).style(
            grid=[2], height="auto"
        )
        text.submit(infer, inputs=[text, iteration_slider, model_selector], outputs=gallery)
        btn.click(infer, inputs=[text, iteration_slider, model_selector], outputs=gallery)
 
    gr.Markdown(
        """___
   <p style='text-align: center'>
   Created by CompVis and Stability AI
   <br/>
   </p>"""
    )
 
demo.launch(debug=True)
```