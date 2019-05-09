import os.path
from radiomics import featureextractor
import nrrd


from img_reader import IMGReader


folder = "/data/images/"
format = "nrrd/auh/"

image_name = format + "auh_image_0.nrrd"
label_name = format + "auh_label_0.nrrd"

img_path = os.getcwd() + folder + image_name
label_path = os.getcwd() + folder + label_name


# image = IMGReader.read_image(img_path)

data, header = nrrd.read(img_path)

extractor = featureextractor.RadiomicsFeatureExtractor()

print("Extraction parameters:\n\t", extractor.settings)
# print("Enabled filters:\n\t", extractor._enabledImagetypes)
# print("Enabled features:\n\t", extractor._enabledFeatures)
#
result = extractor.execute(img_path, label_path)

print("Calculated features:")
for key, value in result.items():
    print("\t", key, ":", value)