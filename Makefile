install_package:
	@pip uninstall -y ddd || :
	@pip install -e .

preprocess_data:
	@rm -rf ${PROCESS_DATA_PATH}/${TO_PREPROCESS}
	@python ddd/ml_logic/preprocess.py ${TO_PREPROCESS};\
