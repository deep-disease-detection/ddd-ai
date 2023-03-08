install_package:
	@pip uninstall -y ddd || :
	@pip install -e .

preprocess_data:
	@rm -rf data/dataset-processed
	@python ddd/ml_logic/preprocess.py ${TO_PREPROCESS};\
