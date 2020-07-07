import functools
from cgan.trainer import CGanTrainer
import torch
import dcgan.dcgan
from os.path import join
from PCA_approach.discriminator import gui_init
from matplotlib.pyplot import imsave
import random
from cgan.combiner import combine_parts

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, current_app
)

PATH_TO_STATIC = 'ui/flask_emoji/static'
PATH_TO_WEIGHTS = 'ui/flask_emoji/pretrained_models'
SEGMENT_LABELS = {"ears": 1, "eyebrows": 2, "eyes": 3, "hands": 4, "mouth": 5, "tears": 6}

bp = Blueprint('index', __name__, url_prefix='/index')


@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        # On Button click generate images

        device = 'cpu'  # CPU should be sufficient for inference in a single image
        if "cgan_gen" in request.form:
            # CGAN Words inference
            embedding_dim = 300
            caption = request.form['caption']
            cgan = CGanTrainer(dataset=None, embedding_dim=embedding_dim, device=device, batch_size=1)
            cgan.load_model(join(PATH_TO_WEIGHTS, "cgan", "word", "gen_weights700.pt"),
                            join(PATH_TO_WEIGHTS, "cgan", "word","dis_weights700.pt"), map_location=device)
            cgan.inference([caption],
                           output_path=join(PATH_TO_STATIC, "cgan_image.png"),
                           glove_model=current_app.config['GloveModel'])
            return render_template('/index.html', cgan_image="cgan_image.png")
        elif "non_conditional_gen" in request.form:
            # DCGAN inference
            dcgan.dcgan.inference(x_in=None, output_path=join(PATH_TO_STATIC, "dcgan_image.png"),
                                  weights=join(PATH_TO_WEIGHTS, "dcgan", "gen_weights.pt"), map_location=device)
            # PCA GAN inference
            seed = random.randint(0, 1000)
            image = gui_init(join(PATH_TO_WEIGHTS, "pcagan", "0.00341766_g6.78450571.pt"), seed=seed,
                             pca_init=current_app.config["PCA.init"])[0]
            imsave(join(PATH_TO_STATIC, 'pcagan_image.png'), image)
            return render_template('/index.html', dcgan_image="dcgan_image.png", pcagan_image="pcagan_image.png")
        elif "segmentgan_gen" in request.form:
            # Segment GAN inference
            embedding_dim = 6
            captions = [SEGMENT_LABELS[key] for key in SEGMENT_LABELS.keys() if key in request.form]
            cgan = CGanTrainer(dataset=None, embedding_dim=embedding_dim, device=device, batch_size=1)
            cgan.load_model(join(PATH_TO_WEIGHTS, "cgan", "segment", "gen_weights516.pt"),
                            join(PATH_TO_WEIGHTS, "cgan", "segment", "dis_weights516.pt"), map_location=device)
            cgan.inference(captions, mode='segment',
                           output_path=join(PATH_TO_STATIC, "segmentgan_image.png"))
            combine_parts(captions, nogan=True, output_path=join(PATH_TO_STATIC, "randsegments_image.png"))

            return render_template('/index.html', segmentgan_image= "segmentgan_image.png",
                                   randsegments_image="randsegments_image.png")
        else:
            raise ValueError("Unknown form")
    return render_template('/index.html')

@bp.route('/model_descriptions')
def model_descriptions():
    return render_template('/model_descriptions.html')

