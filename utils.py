
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

def alpha_to_mask(img):
    alpha = img.split()[-1]
    bg = Image.new("RGBA", img.size, (0,0,0,255))
    bg.paste(alpha, mask=alpha)
    bg = ImageOps.invert(bg.convert('RGB'))
    img = img.convert('RGB')
    return (img, bg)

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
        'generator': generator,
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
