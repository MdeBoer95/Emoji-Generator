from cgan.embeddings.glove_loader import GloveModel
from PCA_approach.discriminator import pca_init

from flask import Flask

def create_app(test_config=None):
    app = Flask(__name__)
    app.config['EmojiFolder'] = "static"
    # Pre load glove model as it takes too much time to load it during inference
    glove_model = GloveModel()
    glove_model.load("cgan/embeddings/glove.6B.300d.txt")
    app.config['GloveModel'] = glove_model
    # PCA
    n_components = 120
    ev, mean_face, pca = pca_init(n_components)
    app.config['PCA.init'] = (ev, mean_face, pca)

    @app.after_request
    def add_header(r):
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
        return r

    @app.route('/')
    def hello_world():
        return 'Hello World'

    from . import index
    app.register_blueprint(index.bp)

    return app
