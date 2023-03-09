install_package:
	@pip uninstall -y ddd || :
	@pip install -e .

preprocess_data:
	@rm -rf ${PROCESS_DATA_PATH}/${TO_PREPROCESS}
	@python -c 'from ddd.ml_logic.preprocess import preprocess_viruses; preprocess_viruses()'

augment_data:
	@python -c 'from ddd.ml_logic.preprocess import augment_pictures; augment_pictures()'


#setting directory for saving parameters localy
create_local_files:
	@mkdir training_outputs
	@mkdir training_outputs/metrics
	@mkdir training_outputs/models
	@mkdir training_outputs/params
