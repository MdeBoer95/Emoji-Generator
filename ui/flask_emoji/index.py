import functools
from cgan.trainer import CGanTrainer
import torch


from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, current_app
)
from werkzeug.security import check_password_hash, generate_password_hash
from cgan.trainer import five, five_string

bp = Blueprint('index', __name__, url_prefix='/index')


@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        # On Button click generate images
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        embedding_dim = 300
        caption = request.form['caption']
        cgan = CGanTrainer(dataset=None, embedding_dim=embedding_dim, device=device, batch_size=5)
        cgan.load_model("/home/marcel/Downloads/gen_weights700.pt", "/home/marcel/Downloads/dis_weights700.pt", map_location=device)
        cgan.inference([caption],
                       output_path="/home/marcel/Uni/Master/3.Semester/DGM/Emoji-Gen/Emoji-Generator/ui/flask_emoji/static/pca_image.png",
                       glove_model=current_app.config['GloveModel'])
        return render_template('/index.html', pca_image="pca_image.png")
    return render_template('/index.html')
