
from fastapi import FastAPI
from transformers import AutoModel

app = FastAPI()

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = AutoModel.from_pretrained("JeffreyXiang/TRELLIS-image-large")

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/model-loaded")
async def is_model_loaded():
    return {"model_loaded": model is not None}

import os
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Set environment variables
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

# Load the pipeline from Hugging Face model hub
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Define a new endpoint to generate 3D models
@app.post("/generate_3d_model")
async def generate_3d_model(image_path: str)
    # Load the image
    image = Image.open(image_path)
    
    # Run the pipeline to generate 3D assets
    outputs = pipeline.run(image, seed=1)

    # Handle the outputs and save them
    video_gaussian = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave("output_sample_gs.mp4", video_gaussian, fps=30)

    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave("output_sample_mesh.mp4", video_mesh, fps=30)

    # Export GLB and PLY files
    glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.95, texture_size=1024)
    glb.export("output_sample.glb")
    outputs['gaussian'][0].save_ply("output_sample.ply")

    return {"message": "3D model generated successfully!", "files": ["output_sample.glb", "output_sample.ply", "output_sample_gs.mp4", "output_sample_mesh.mp4"]}
