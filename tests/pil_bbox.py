from PIL import Image
from PIL import ImageDraw

before = Image.open('../data/raw/mask/demo/mask.png').convert('L')
after = before.copy()
ImageDraw.Draw(after).rectangle(before.getbbox(), fill='white')

before.show()
after.show()
