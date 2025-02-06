## IBM/NASA Prithvi Geospatial model demonstrator on VLLM


This repository provides a demo application for the Prithvi Geospatial, developed in collaboration between IBM and NASA,
using vLLM as hte model inference engine.
This is an adaptation the original demo available [here](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Sen1Floods11-demo).

### Requirements:

<u>vLLM docker image that includes the Prithvi model</u>

To build a vLLM docker image please use the vLLM source base available [here](https://github.com/christian-pinto/vllm/tree/ibm_prithvi_geospatial)


<u>Prithvi model weights</u>

The file [Prithvi-EO-V2-300M-TL-Sen1Floods11.pt](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/tree/main) is to be downloaded and placed inside the `model` folder of this repository.

### Running the demo:

```shell
docker run --rm -it -p 7860:7860  \ 
       -v  /path/to/this/repository:/workspace/scripts -e VLLM_CONFIGURE_LOGGING=0 \
       --entrypoint python3 your-vllm-image-tag scripts/demo_flooding/app.py

```

The demo starts a web server listening on `http://localhost:7860/`. Open the URL in your browser and run the demo.

