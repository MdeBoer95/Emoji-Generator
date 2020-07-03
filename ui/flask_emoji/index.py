import functools
from cgan.trainer import CGanTrainer
import torch
import dcgan.dcgan
from os.path import join


from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, current_app
)
from werkzeug.security import check_password_hash, generate_password_hash
from cgan.trainer import five, five_string

PATH_TO_STATIC = 'ui/flask_emoji/static'
PATH_TO_WEIGHTS = 'ui/flask_emoji/pretrained_models'

bp = Blueprint('index', __name__, url_prefix='/index')


@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        # On Button click generate images

        device = 'cpu'  # CPU should be sufficient for inference in a single image
        if "cgan_gen" in request.form:
            embedding_dim = 300
            caption = request.form['caption']
            cgan = CGanTrainer(dataset=None, embedding_dim=embedding_dim, device=device, batch_size=1)
            cgan.load_model(join(PATH_TO_WEIGHTS, "cgan", "gen_weights700.pt"),
                            join(PATH_TO_WEIGHTS, "cgan", "dis_weights700.pt"), map_location=device)
            cgan.inference([caption],
                           output_path=join(PATH_TO_STATIC, "cgan_image.png"),
                           glove_model=current_app.config['GloveModel'])
            return render_template('/index.html', cgan_image="cgan_image.png")
        elif "non_conditional_gen":
            dcgan.dcgan.inference(x_in=None, output_path=join(PATH_TO_STATIC, "dcgan_image.png"),
                                  weights=join(PATH_TO_WEIGHTS, "dcgan", "gen_weights.pt"), map_location=device)
            return render_template('/index.html', dcgan_image="dcgan_image.png")
        else:
            raise ValueError("Unknown form")
    return render_template('/index.html')
