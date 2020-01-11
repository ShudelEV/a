import falcon
from .resources import check_one_object

__version__ = "0.1.0"

app = falcon.API()
app.add_route('/object/', check_one_object)
