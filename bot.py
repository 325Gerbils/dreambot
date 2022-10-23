
from __future__ import unicode_literals
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
import subprocess
import gc
import re
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers as diffusers
import math
import youtube_dl
import whisper
from pipeline.unified_pipeline import UnifiedPipeline

model_name = "./lyra-diffusion-v1-5"
with open('loaded_model.txt', 'r') as f:
    model_name = f.read()
    f.close()

pipelines = {
    "text2img": "StableDiffusionPipeline",
    "img2img": "StableDiffusionImg2ImgPipeline",
    "inpaint": "StableDiffusionInpaintPipeline",
    "unified": "UnifiedPipeline"
}
print(UnifiedPipeline)

# FUNCTIONS ------------------------------------------------------------------------

def restart_program():
    SCRIPT_FILE_NAME = os.path.basename(__file__)
    print("Restarting...")
    run_string = 'python "'+ SCRIPT_FILE_NAME + '"'
    print(run_string)
    subprocess.Popen(run_string)
    exit(0)

def autoresize(img, max_p):
    w, h = img.size
    total_pixels = w * h
    if total_pixels > max_p:
        ratio = 600 / math.sqrt(total_pixels)
        w, h = map(lambda x: int(x * ratio), (w, h))
        w, h = map(lambda x: x - x % 64, (w, h))
        print(w, h)
        img = img.resize( (w, h) )
    return img

def load_pipeline(model):
    global loaded_model, pipe

    # if loaded_model == model:
    #     return f'model {model} already loaded'
    if model not in pipelines:
        return f"model {model} not found: try using 'text2img', 'img2img', or 'inpaint'"

    global unet, scheduler, vae, text_encoder, tokenizer
    clear_cuda_memory()
    if model == 'unified':
        pipeline = UnifiedPipeline
    else:
        pipeline = getattr(diffusers, pipelines[model])
    pipe = pipeline.from_pretrained(
        model_name,
        unet=unet,
        scheduler=scheduler,
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
    pipe = None
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

def call_gpt3(prompt, temp=0.8, tokens=100):
    response = openai.Completion.create(model='text-davinci-002', prompt=prompt, temperature=temp, max_tokens=tokens)
    if response.choices[0].text.strip() == '':
        return call_gpt3(prompt, temp=temp, tokens=tokens)
    return response.choices[0].text
    
def prompt_enhance(kwords, temp=0.8, tokens=50):
    print('asking gpt3 to enhance...')
    # enhance is designed to take a couple of words and write a sentence or two full of visually descriptive text
    # it also works well to juice up a regular prompt instead of just using keywords
    # it likes to be grammatically correct more than it likes to spam keywords like stable diffusion prompters lol
    kwords_prompt = """Take a couple of keywords and write a bunch of visually descriptive text for each one.\n
Here are some examples:\n
keywords: sunrise meadow clouds\n
prompt: a beautiful sunrise in a meadow, surrounded by clouds, beautiful painting, by rebecca guay, yoshitaka amano, trending on artstation hq\n\n
keywords: lawyer\n
prompt: a beefy intimidating copyright lawyer with glowing red eyes, dramatic portrait, digital painting portrait, ArtStation\n\n
keywords: marshmallow eiffel\n
prompt: giant pink marshmallow man stomping towards the eiffel tower, film scene, design by Pixar and WETA Digital, 8k photography\n\n
keywords: hat guitar ocean\n
prompt: a black and white photograph of a man wearing a hat playing a guitar in the ocean, hd 4k hyper detailed\n\n
keywords: girl plants ghibli\n
prompt: a girl watering her plants, studio ghibli, art by hayao miyazaki, artstation hq, wlop, by greg rutkowski, ilya kuvshinov \n\n
keywords: psychedelic\n
prompt: ego death, visionary artwork by Alex Grey, hyperdetailed digital render, fractals, dramatic, 3d render\n\n
Make sure to include style annotations in your prompt, such as '8k photograph', 'digital art', 'oil painting', 'realistic, detailed', 'portrait', and 'by '\n
Make a prompt for the following keywords:\n
keywords: """+kwords+"""\n
prompt: """
    response = call_gpt3(kwords_prompt).strip()
    # response = response.split("keywords:")[0] if 'keywords:' in response else response
    if response == '':
        return prompt_enhance(kwords, temp=temp, tokens=tokens)
    return response

def get_positivity(prompt):
    print('asking gpt3 to figure out the positivity...')
    positivity_prompt = f"""You reply with True or False depending on a given words positivity.
Interpret everything with a hint of sarcasm.
Here are some examples:
on
True
off
False
1
True
0
False
yes
True
no
False
done
True
thats all folks
False
totally dude
True
im good thanks
False
Now it's your turn:
{prompt}"""
    response = call_gpt3(positivity_prompt, temp=0.5, tokens=10)
    print(f'response "{response}"')
    parsed_response = 'True' in response
    print(f'parsed as "{parsed_response}"')
    return parsed_response

def get_facts_from_text(text):
    facts_prompt = f"""You determine whether any facts in a given text would be useful to remember. 
Here are some examples of useful facts: <username>'s name is paul, i am an AI model, you frequently say dang.
If there are any, reply with them rewritten in a concise form. 
If there are multiple, separate them with commas. 
If there are none, write 'None'.\nHere is the given text:\n {text}"""
    response = call_gpt3(facts_prompt)
    print(f'facts response "{response}"')
    return response

def combine_prompts_gpt(prompt1, prompt2):
    combine_prompt = f"""You are given two text prompts. Your job is to write a single prompt based on a combination of the two given prompts. combine the following two prompts:
1) {prompt1}
2) {prompt2}
combination:"""
    response = call_gpt3(combine_prompt)
    print(f'facts response "{response}"')
    return response

import traceback
def call_stable_diffusion(prompt, kwargs):
    try:
        kwargs = {
            'generator': kwargs['generator'],
            'strength': float(kwargs['strength']) if 'strength' in kwargs else 0.75,
            'num_inference_steps': int(kwargs['steps']) if 'steps' in kwargs else 50,
            'guidance_scale': float(kwargs['scale']) if 'scale' in kwargs else 7.5,
            'height': int(kwargs['height']) if 'height' in kwargs else 512,
            'width': int(kwargs['width']) if 'width' in kwargs else 512,
            'init_image': kwargs['init_image'].convert('RGB') if 'init_image' in kwargs else None,
            'mask_image': kwargs['mask_image'].convert('RGB') if 'mask_image' in kwargs else None,
            'negative_prompt': kwargs['negative_prompt'] if 'negative_prompt' in kwargs else '',
            'outmask_image': kwargs['mask_image'] if 'outmask' in kwargs else None
        }
        if 'outmask' in kwargs:
            del kwargs['outmask']
        with autocast("cuda"), torch.inference_mode():
            image = pipe(prompt, **kwargs).images[0]
    except:
        traceback.print_exc()
    return image

def PIL_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGBA')

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

def save_audio(url, folder):
    gen_count = len(os.listdir(f'{folder}'))
    audio_fname = f'{folder}/{gen_count}.{url.split(".")[-1]}'
    response = requests.get(url)
    audiofile = response.content.strip()
    with open(audio_fname, 'wb') as f:
        f.write(audiofile)
        f.close()
    return audio_fname

def separate_alpha_to_inpaint_mask(img):
    mask = Image.new("RGBA", img.size, (255,255,255,255))
    black = Image.new("RGBA", img.size, (0,0,0,255))
    mask.paste(black, mask=img.split()[-1])
    return img.convert('RGB'), mask

# END FUNCTIONS ------------------------------------------------------------------------

print('starting...')
intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(command_prefix="-", intents=intents)

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'
TOKEN = os.getenv('DISCORD_TOKEN')
openai.api_key = os.getenv('OPENAI_TOKEN')
random.seed(time.time())

# whisper_model = whisper.load_model("small")

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
offload_device = "cpu"

try:
    vae = diffusers.AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    unet = diffusers.UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # scheduler = diffusers.LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

except Exception as e:
    print(f'error loading model: {e}')
    print('loading stable-diffusion-v1-5')
    model_name = "./stable-diffusion-v1-5"
    with open('loaded_model.txt', 'w') as f:
        f.write(model_name)
        f.close()
    restart_program()

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

seamless = False
pipe = None
pipe = load_pipeline('unified')
last_used = time.time()

@bot.command()
async def dream(ctx, *prompt):
    global loaded_model, pipe, last_used

    # prompt/attachment parsing
    prompt, kwargs = parse_prompt(prompt)
    n_images = int(kwargs['n']) if 'n' in kwargs else 1
    enhance = True if 'enhance' in kwargs else False
    if '[' in prompt:
        negative_prompt = prompt.split('[')[1].split(']')[0]
        prompt = "".join([prompt.split('[')[0], prompt.split(']')[1]])
        kwargs['negative_prompt'] = negative_prompt
    else:
        negative_prompt = ''   

    if len(ctx.message.attachments) > 0:
        input_img = PIL_from_url(ctx.message.attachments[0].url)
        input_img = autoresize(input_img, 250_000)
        kwargs['init_image'] = input_img
        if len(ctx.message.attachments) > 1:
            # use 2nd attachment as mask
            mask = PIL_from_url(ctx.message.attachments[1].url)
            mask = autoresize(mask, 250_000)
            kwargs['mask_image'] = mask
    
    # generation loop
    for i in range(n_images):
        seed = int(kwargs['seed']) if 'seed' in kwargs else random.randrange(0, 2**32)
        kwargs['generator'] = torch.Generator("cuda").manual_seed(seed)
        if enhance:
            t = float(kwargs['temp']) if 'temp' in kwargs else 0.75
            print(t)
            prompt = prompt_enhance(prompt, temp=t)
        print(f'\"{prompt}\" by {ctx.author.name}, {kwargs} ({i+1}/{n_images})')
        await ctx.send(f"starting dream for `{prompt}{'' if negative_prompt == '' else f'[{negative_prompt}]'}` with seed {seed} ({i+1}/{n_images})")
        
        start_time = time.time()
        try:
            image = call_stable_diffusion(prompt, kwargs)
        except Exception as e:
            await ctx.send(f'{e}')
        
        # save and send
        filename = save_img(image, 'outputs')
        elapsed_time = int(time.time() - start_time)
        last_used = time.time()
        await ctx.send(f"\"{prompt}{'' if negative_prompt == '' else f'[{negative_prompt}]'}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})", file=discord.File(filename))
        
@bot.command()
async def palette(ctx, *prompt):
    global loaded_model, pipe, last_used

    # prompt parsing
    prompt, kwargs = parse_prompt(prompt)
    n_colors = math.floor(float(kwargs['colors'])) if 'colors' in kwargs else 5
    n_colors = (n_colors if n_colors > 3 else 4) if n_colors < 8 else 7
    n_images = int(kwargs['n']) if 'n' in kwargs else 1
    if loaded_model not in ['text2img', 'unified']:        
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
    input_img = PIL_from_url(ctx.message.attachments[0].url).convert('RGB')
    import torch
    from esr.realesrgan import RealESRGAN
    device1 = torch.device('cuda')
    esr_model = RealESRGAN(device1, scale=4)
    esr_model.load_weights('esr\weights\RealESRGAN_x4.pth')
    upscaled_img = esr_model.predict(input_img)
    path = save_img(upscaled_img, 'outputs/upscaled')
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
    if conceptname == '':
        await ctx.send('go to `huggingface.co/sd-concepts-library` to find concepts')
        return
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
    await ctx.send(f'Loading stable diffusion {model} pipeline...')
    pipe = load_pipeline(model)
    await ctx.send(f'{model} model loaded. happy generating!')

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
    await ctx.send(prompt_enhance(' '.join(prompt)))

# broken
@bot.command()
async def seamless(ctx, *prompt):
    global pipe, loaded_model, seamless
    if prompt == ():
        await ctx.send(f'seamless is currently {"on" if seamless else "off"}')
        return
    positive = get_positivity(' '.join(prompt))
    if positive:
        if seamless:
            await ctx.send('seamless is already on')
            return
        seamless = True
        await ctx.send('turning seamless on...')
        reset_conv()
        patch_conv(padding_mode='circular')
        pipe = load_pipeline(loaded_model)
    if not positive:
        if not seamless:
            await ctx.send('seamless is already off')
            return
        seamless = False
        await ctx.send('turning seamless off...')
        reset_conv()
        patch_conv(padding_mode='zeros')
        pipe = load_pipeline(loaded_model)
    await ctx.send(f'reloaded {loaded_model} with seamless {"on" if seamless else "off"}')

@bot.command()
async def positive(ctx, *prompt):
    positive = get_positivity(' '.join(prompt))
    await ctx.send(f'i think that\'s {"positive" if positive else "negative"}')

@bot.command()
async def invert(ctx, *prompt):
    image = PIL_from_url(ctx.message.attachments[0])
    image = ImageOps.invert(image)
    path = save_img(image, 'outputs/inverted')
    print('done')
    await ctx.send(f'inverted', file=discord.File(path))

@bot.command()
async def finetune(ctx, *prompt):
    prompt, kwargs = parse_prompt(prompt)
    images = [PIL_from_url(ctx.message.attachments[i]) for i in range(len(ctx.message.attachments))]
    n_out_folders = sum([len(folder) for r, d, folder in os.walk('inputs/finetuning')])
    outfolder_path = f'inputs/finetuning/{n_out_folders}'
    if not os.path.exists(outfolder_path):
        os.makedirs(outfolder_path)
    for image in images:
        save_img(image, outfolder_path)
    with open(f'{outfolder_path}/prompt.txt', 'w') as f:
        f.write(prompt)
        f.close()
    print('done')
    await ctx.send(f'finished saving images for `{prompt}`, will finetune on colab next opportunity i get')
    
@bot.command()
async def combine_prompts(ctx, *prompt):
    prompt1, prompt2 = ' '.join(prompt).split(';')
    combined = combine_prompts_gpt(prompt1, prompt2)
    await ctx.send(f'{combined}')

@bot.command()
async def ytmp3(ctx, url):
    await ctx.send(f'downloading mp3 from `{url}`...')
    gen_count = len(os.listdir('outputs/mp3s'))
    path = f'outputs/mp3s/{gen_count}.mp3'
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': path,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        info_dict = ydl.extract_info(url, download=False)
        video_title = info_dict.get('title', None)
        await ctx.send(f'`{video_title}`.mp3', file=discord.File(path))
  
@bot.command()
async def whisper(ctx):
    print('starting whisper...')
    await ctx.send('whisper is transcribing...')
    path = save_audio(ctx.message.attachments[0].url, 'inputs/whisper')
    result = whisper_model.transcribe(path)["text"].strip()
    await ctx.send(f'> {result}')
    print('done')

@bot.command()
async def change_model(ctx, model_name):
    if model_name not in ['lyra2', 'sus', 'stable'] and ctx.message.author.id != 891221733326090250:
        await ctx.send(f'cannot find {model_name} in finetunes. try "stable", "sus", or "lyra2"')
        return
    model_name = f"./{model_name}-diffusion-v1-5"
    with open('loaded_model.txt', 'w') as f:
        f.write(model_name)
        f.close()
    await ctx.send(f'reloading with {model_name}')
    restart_program()

@bot.command()
async def restart(ctx):
    await ctx.send('restarting... please wait a minute or two')
    restart_program()

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
