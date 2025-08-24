import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

FILE_PATH_TRAIN_POLYGONS = os.path.join(BASE_DIR, "data", "gdf_train_local.geojson")
FILE_PATH_TEST_POLYGONS  = os.path.join(BASE_DIR, "data", "gdf_test_local.geojson")

DIR_TRAIN_ORTHOPHOTOS = os.path.join(BASE_DIR, "data", "train_orthophotos")
DIR_TEST_ORTHOPHOTOS  = os.path.join(BASE_DIR, "data", "test_orthophotos")

MODELS_PATH = os.path.join(BASE_DIR, "models")
