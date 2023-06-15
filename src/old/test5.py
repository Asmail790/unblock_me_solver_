from tools.grid.utils.implementions import ConvertImageToGrid,BoundingBox,BoundingBox
from tools.grid.utils.convience import draw_puzzle


blocksprops= [
    BoundingBox(clazz=2, left=208.801513671875, right=362.7470703125, top=748.9385375976562, bottom=1073.9852294921875),
    BoundingBox(clazz=2, left=713.0460205078125, right=868.8106689453125, top=920.1548461914062, bottom=1240.9617919921875),
    BoundingBox(clazz=2, left=378.9742431640625, right=532.390380859375, top=1084.0621337890625, bottom=1408.79248046875),
    BoundingBox(clazz=2, left=39.6920166015625, right=193.96563720703125, top=1422.834716796875, bottom=1751.9080810546875),
    BoundingBox(clazz=5, left=27.17529296875, right=375.42864990234375, top=1086.296875, bottom=1246.641845703125),
    BoundingBox(clazz=1, left=539.8094482421875, right=1049.072021484375, top=1417.979736328125, bottom=1586.494384765625) 
]
gridprops = BoundingBox(left=28.43414306640625, right=1054.776123046875, top=734.7093505859375, bottom=1766.4945068359375)


view = draw_puzzle(ConvertImageToGrid().convert(blocksprops,gridprops))
print(view)
