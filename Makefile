install_package:
	@pip uninstall -y ddd || :
	@pip install -e .

preprocess_data:
	@python -c 'from ddd.ml_logic.preprocess import preprocess_viruses; preprocess_viruses()'

augment_data:
	@python -c 'from ddd.ml_logic.preprocess import augment_pictures; augment_pictures()'

train_model:
	@python -c 'from ddd.interface import train_model; train_model()'
