
import discord
from discord.ext import commands

import os, time, random
from dotenv import load_dotenv
from PIL import Image, ImageOps
import requests
from io import BytesIO
from colorthief import ColorThief
import numpy as np
import torch
import openai
from torch import autocast
import gc
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers as diffusers
import math

print('starting...')
intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(command_prefix="-", intents=intents)

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
openai.api_key = os.getenv('OPENAI_TOKEN')
random.seed(time.time())

model_name = "./stable-diffusion-v1-4"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
offload_device = "cpu"

vae = diffusers.AutoencoderKL.from_pretrained(model_name, subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
unet = diffusers.UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
scheduler = diffusers.LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

vae = vae.to(offload_device).half()
text_encoder = text_encoder.to(offload_device).half()
unet = unet.to(torch_device).half()

requires_grad(vae)
requires_grad(text_encoder)
requires_grad(unet)

print('loading concepts....')
for concept in os.listdir('concepts'):
    token_name = f'<{concept.split(".")[0]}>'
    print(f'loading {token_name}')
    text_encoder, tokenizer = load_learned_embed_in_clip(f'concepts/{concept}', text_encoder, tokenizer, token_name)

loaded_model = 'None'
pipelines = {
    "text2img": {
        "pipeline": "StableDiffusionPipeline",
        "pad_mode": "zeros"
    },
    "img2img": {
        "pipeline": "StableDiffusionImg2ImgPipeline",
        "pad_mode": "zeros"
    },
    "inpaint": {
        "pipeline": "StableDiffusionInpaintPipeline",
        "pad_mode": "zeros"
    },
    "seamless": {
        "pipeline": "StableDiffusionPipeline",
        "pad_mode": "circular"
    },
    "img2img_seamless": {
        "pipeline": "StableDiffusionImg2ImgPipeline",
        "pad_mode": "circular"
    },
}

pipe = load_pipeline('text2img')
last_used = time.time()
print('ready to go!')

@bot.command()
async def dream(ctx, *prompt):
    global loaded_model, pipe, last_used

    # prompt/attachment parsing
    prompt, kwargs = parse_prompt(prompt)
    n_images = int(kwargs['n']) if 'n' in kwargs else 1
    if len(ctx.message.attachments) > 0:
        input_img = PIL_from_url(ctx.message.attachments[0].url)
        input_img = autoresize(input_img, 380000)
        kwargs['init_img'] = input_img
        if len(ctx.message.attachments) > 1:
            mask = PIL_from_url(ctx.message.attachments[1].url)
            kwargs['mask'] = mask
    
    # figure out pipeline
    if len(ctx.message.attachments) < 1 and loaded_model not in ['inpaint']:
        await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model inpaint` and try again.')
        return
    if len(ctx.mssage.attachments) > 0 and loaded_model not in ['inpaint', 'img2img']:
        await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model img2img` and try again.')
        return
    if len(ctx.mssage.attachments) < 0 and loaded_model not in ['seamless', 'text2img']:
        await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model text2img` and try again.')
        return
    
    # generation loop
    for i in range(n_images):
        seed = int(kwargs['seed']) if 'seed' in kwargs else random.randrange(0, 2**32)
        kwargs['generator'] = torch.Generator("cuda").manual_seed(seed)
        print(f'\"{prompt}\" by {ctx.author.name}')
        await ctx.send(f"starting dream for `{prompt}` with seed {seed} ({i+1}/{n_images})")
        
        start_time = time.time()
        image = call_stable_diffusion(prompt, kwargs)
        
        # save and send
        filename = save_img(image, 'outputs')
        elapsed_time = int(time.time() - start_time)
        last_used = time.time()
        await ctx.send(f"\"{prompt}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})", file=discord.File(filename))
        
@bot.command()
async def palette(ctx, *prompt):
    global loaded_model, pipe, last_used

    # prompt parsing
    prompt, kwargs = parse_prompt(prompt)
    n_colors = math.floor(float(kwargs['colors'])) if 'colors' in kwargs else 5
    n_colors = (n_colors if n_colors > 3 else 4) if n_colors < 8 else 7
    n_images = int(kwargs['n']) if 'n' in kwargs else 1
    if loaded_model not in ['text2img']:        
        await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model text2img` and try again.')
        return

    # generation loop
    for i in range(n_images):
        seed = int(kwargs['seed']) if 'seed' in kwargs else random.randrange(0, 2**32)
        kwargs['generator'] = torch.Generator("cuda").manual_seed(seed)
        await ctx.send(f"making color palette for `{prompt}` with seed {seed}")

        # make image with SD
        print('starting...')
        start_time = time.time()
        image = call_stable_diffusion(prompt, kwargs)
        filename = save_img(image, 'outputs')

        # get palette from image
        color_thief = ColorThief(filename)
        palette = color_thief.get_palette(color_count=n_colors, quality=10)
        hex_colors = ' '.join(f'#{rgb_to_hex(v).upper()}' for v in palette)
        palette = np.uint8(palette).reshape(1, n_colors, 3)
        pal_img = Image.fromarray(palette).resize((64*n_colors, 64), Image.Resampling.NEAREST)

        # save and send
        pal_img_fname = save_img(pal_img, 'outputs/palettes')
        thumb = image.resize((64, 64))
        thumb_fname = save_img(thumb, 'outputs/thumbs')
        elapsed_time = int(time.time() - start_time)
        await ctx.send(f"color palette for \"{prompt}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})\n{hex_colors}", files=[discord.File(pal_img_fname), discord.File(thumb_fname)])

@bot.command()
async def upscale(ctx, *prompt):
    print('upscaling')
    input_img = PIL_from_url(ctx.message.attachments[0].url)
    import torch
    from esr.realesrgan import RealESRGAN
    device1 = torch.device('cuda')
    esr_model = RealESRGAN(device1, scale=4)
    esr_model.load_weights('esr\weights\RealESRGAN_x4.pth')
    upscaled_img = esr_model.predict(input_img)
    path = save_img(upscaled_img, 'upscaled')
    print('done')
    await ctx.send(f"upscaled by {ctx.author.mention}", file=discord.File(path))

@bot.command()
async def tile(ctx, *prompt):
    # parse prompt and attachment
    prompt, kwargs = parse_prompt(prompt)
    target_x = int(kwargs['x']) if 'x' in kwargs else 4
    target_y = int(kwargs['y']) if 'y' in kwargs else 4
    input_img = PIL_from_url(ctx.message.attachments[0].url)

    # make target canvas
    w, h = int(input_img.size[0]/2), int(input_img.size[1]/2)
    input_img = input_img.resize((w, h))
    img_collage = Image.new('RGB', (w*target_x, h*target_y))

    # paste the attachment onto canvas
    for i in range(target_x):
        for j in range(target_y):
            img_collage.paste(input_img, (i*w, j*h))

    # save and send
    tiled_fname = save_img(img_collage, 'outputs/tiled')
    await ctx.send(f'tiled by {ctx.author.mention}', file=discord.File(tiled_fname))

@bot.command()
async def get_concept(ctx, conceptname):
    global tokenizer, text_encoder
    downloadurl = f'https://huggingface.co/sd-concepts-library/{conceptname}/resolve/main/learned_embeds.bin'
    print(downloadurl)
    response = requests.get(downloadurl)
    with open(f'concepts/{conceptname}.bin', 'bw') as f:
        f.write(response.content)
        f.close()
    token_name = f'<{conceptname.split(".")[0]}>'
    print(f'loading {token_name}')
    text_encoder, tokenizer = load_learned_embed_in_clip(f'concepts/{conceptname}.bin', text_encoder, tokenizer, token_name)
    await ctx.send(f'finished downloading `<{conceptname}>` from `huggingface.co/sd-concepts-library`')

@bot.command()
async def load_model(ctx, model):
    global pipe
    await ctx.send('Loading stable diffusion '+model+' pipeline...')
    pipe = load_pipeline(model)
    await ctx.send(model+' model loaded. happy generating!')

@bot.command()
async def clear_cuda_mem(ctx):
    clear_cuda_memory()
    await ctx.send('cuda cleared')

@bot.command()
async def last_used(ctx):
    global last_used
    since = int(time.time() - last_used)
    await ctx.send(f'last used {since}s ago')

@bot.command()
async def gpt(ctx, *prompt):
    await ctx.send(call_gpt3(' '.join(prompt)))

@bot.command()
async def enhance(ctx, *prompt):
    await ctx.send(kprompt_call(' '.join(prompt)))

print("Connected")
bot.run(TOKEN)

# @bot.command()
# async def interrogate(ctx):
#     await ctx.send('starting clip-interrogator...')
#     print('Interrogating...')
#     from clip_interrogator.clip_interrogator import interrogate
#     response = requests.get(ctx.message.attachments[0].url)
#     input_img = Image.open(BytesIO(response.content)).convert('RGB')
#     response = interrogate(input_img, models=['ViT-L/14'])
#     print('done')
#     await ctx.send(f'clip-interrogator thinks your picture looks like `{response}`')


        # # reaction = (<Reaction emoji='🖼️' me=False count=1>, <Member id=891221733326090250 name='bleepybloops' discriminator='6448' bot=False nick='bleep bloop' guild=<Guild id=804209375098568724 name='Creativity Farm' shard_id=0 chunked=False member_count=29>>)
        # def check(reaction, user):
        #     return user == ctx.message.author
        # reaction = await bot.wait_for("reaction_add", check=check)
        # ai_art_channel_id = 907829317575245884
        # if '🖼' in reaction[0].emoji:
        #     print('reacted with 🖼')
        #     await bot.get_channel(ai_art_channel_id).send(f"\"{prompt}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})", file=discord.File(filename))
