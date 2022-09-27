
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

# FUNCTIONS ------------------------------------------------------------------------

def autoresize(img, max_p):
    img_width, img_height = img.size
    total_pixels = img_width*img_height
    if total_pixels > max_p:
        import math
        ratio = 632 / math.sqrt(total_pixels)
        new_size = [64 * int(ratio * s) % 64 for s in img.size]
        img = img.resize( new_size )
        return img
    return img

def load_pipeline(model):
    global loaded_model, pipe

    if loaded_model == model:
        return f'model {model} already loaded'
    if model not in pipelines:
        return f'model {model} not found: try using \'text2img\', \'img2img\', \'inpaint\', or \'seamless\''

    global unet, scheduler, vae, text_encoder, tokenizer
    chosen_pipeline = pipelines[model]
    clear_cuda_memory()
    reset_conv()
    pipeline = getattr(diffusers, chosen_pipeline["pipeline"])
    patch_conv(padding_mode=chosen_pipeline["pad_mode"])
    pipe = pipeline.from_pretrained(
        "./stable-diffusion-v1-4",
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision="fp16",
        torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    torch.manual_seed(0)
    loaded_model = model
    print(f'loaded model: {loaded_model}')
    return pipe

def clear_cuda_memory():
    global pipe, torch
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return text_encoder, tokenizer

def requires_grad(model, val=False):
    for param in model.parameters():
        param.requires_grad = val
    
def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

original_conv2d_init = torch.nn.Conv2d.__init__
def patch_conv(**patch):
    global original_conv2d_init
    cls = torch.nn.Conv2d
    init = cls.__init__
    original_conv2d_init = init
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__

def reset_conv():
    if original_conv2d_init is None:
        print('Resetting convolution failed, no original_conv2d_init')
    cls = torch.nn.Conv2d
    cls.__init__ = original_conv2d_init

def call_gpt3(prompt):
    response = openai.Completion.create(model='text-davinci-002', prompt=prompt, temperature=0.75, max_tokens=100)
    print(response.choices[0].text.strip())
    return response.choices[0].text
    
def kprompt_call(kwords):
    print('asking gpt3 to enhance...')
    # enhance is designed to take a couple of words and write a sentence or two full of visually descriptive text
    # it also works well to juice up a regular prompt instead of just using keywords
    # it likes to be grammatically correct more than it likes to spam keywords like stable diffusion prompters lol
    kwords_prompt = """Take a couple of keywords and write a sentence or two full of visually descriptive text for each one.\n
    Here are some examples:\n
    keywords: hat guitar ocean\n
    prompt: a black and white photograph of a man wearing a hat playing a guitar in the ocean, hd 4k hyper detailed\n\n
    keywords: sunrise meadow clouds\n
    prompt: a beautiful sunrise in a meadow, surrounded by clouds, beautiful painting, by rebecca guay, yoshitaka amano, trending on artstation hq\n\n
    keywords: lawyer\n
    prompt: a beefy intimidating copyright lawyer with glowing red eyes, dramatic portrait, digital painting portrait, ArtStation\n\n
    keywords: marshmallow eiffel\n
    prompt: giant pink marshmallow man stomping towards the eiffel tower, film scene, design by Pixar and WETA Digital, 8k photography\n\n
    keywords: girl plants ghibli\n
    prompt: a girl watering her plants, studio ghibli, art by hayao miyazaki, artstation hq, wlop, by greg rutkowski, ilya kuvshinov \n\n
    keywords: psychedelic\n
    prompt: ego death, visionary artwork by Alex Grey, hyperdetailed digital render, fractals, dramatic, 3d render\n\n
    Make sure to include style annotations in your prompt, such as '8k photograph', 'digital art', 'oil painting', 'realistic, detailed', 'portrait', and 'by '\n
    Now it's your turn! \n
    Make a detailed prompt for the following keywords.\n
    keywords:	"""+kwords+"""\n
    prompt: """
    response = call_gpt3(kwords_prompt)
    # response = response.split("keywords:")[0] if 'keywords:' in response else response
    if response.strip() == '':
        return kprompt_call(kwords)
    print(response)
    return response

def call_stable_diffusion(prompt, kwargs):
    kwargs = {
        'generator': kwargs['generator'],
        'strength': float(kwargs['strength']) if 'strength' in kwargs else 0.75,
        'num_inference_steps': int(kwargs['steps']) if 'steps' in kwargs else 50,
        'guidance_scale': float(kwargs['scale']) if 'scale' in kwargs else 7.5,
        'height': int(kwargs['height']) if 'height' in kwargs else 512,
        'width': int(kwargs['width']) if 'width' in kwargs else 512,
        'init_img': kwargs['init_img'] if loaded_model in ['inpaint', 'img2img', 'img2img_seamless'] else None,
        'mask_image': kwargs['mask_image'] if loaded_model in ['inpaint'] else None
    }
    if kwargs['init_img'] == None:
        del kwargs['init_img']
    if kwargs['mask_image'] == None:
        del kwargs['mask_image']

    with autocast("cuda"):
        image = pipe(prompt, **kwargs).images[0]

    return image

def PIL_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGB')

def parse_prompt(prompt):
    arg_words = [t for t in prompt if '=' in t]
    kwargs = dict(t.split('=') for t in arg_words) if arg_words is not [] else {}
    prompt = ' '.join([t for t in prompt if '=' not in t])
    return prompt, kwargs

def save_img(image, folder):
    gen_count = len(os.listdir(f'{folder}'))
    img_fname = f'{folder}/{gen_count}.png'
    image.save(img_fname)
    return img_fname

# END FUNCTIONS ------------------------------------------------------------------------

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

print('loading concepts...')
for concept in os.listdir('concepts'):
    token_name = f'<{concept.split(".")[0]}>'
    # print(f'loading {token_name}')
    text_encoder, tokenizer = load_learned_embed_in_clip(f'concepts/{concept}', text_encoder, tokenizer, token_name)
    
print('loading pipeline...')
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
pipe = None
pipe = load_pipeline('text2img')
last_used = time.time()

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
            kwargs['mask_image'] = mask
    
    # figure out pipeline
    if len(ctx.message.attachments) == 1 and loaded_model not in ['inpaint']:
        await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model inpaint` and try again.')
        return
    if len(ctx.message.attachments) > 0 and loaded_model not in ['inpaint', 'img2img']:
        await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model img2img` and try again.')
        return
    if len(ctx.message.attachments) <= 0 and loaded_model not in ['seamless', 'text2img']:
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

print("connected, ready to dream!")
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
